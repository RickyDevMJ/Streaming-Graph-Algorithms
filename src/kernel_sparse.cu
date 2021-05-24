#include "kernel_sparse.h"
#include <stdgpu/cstddef.h>
#include <stdgpu/iterator.h>
#include <stdgpu/unordered_set.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   	if (code != cudaSuccess) 
   	{
	  	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  	if (abort) exit(code);
   	}
}

using namespace cooperative_groups;

__global__ void update_colors(int m, int* scc, int* colors){
   	int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   	int scc_id = -1;
   	if(tid < m){
	  	scc_id = scc[tid];
	  	colors[scc_id] = 1;
   	}

   	grid_group grid = this_grid();
   	grid.sync();

   	int tmp;
   	// prefix sum
   	for(int off = 1; off < m; off *= 2){
	  	if(tid >= off && tid < m){
		 	tmp = colors[tid - off];
	  	}
	  	grid.sync();
	  
	  	if(tid >= off && tid < m){
			colors[tid] += tmp;
	 	}
	  	grid.sync();
   	}

   	if(tid < m){
	  	scc[tid] = colors[scc_id] - 1;
   	}
}

__global__ void merge_scc_kernel(int m, int num_scc, int nnz, int* scc, int* colors, int* trc_column, int* trc_row){
   	int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
   	if(tid < num_scc)
	  	colors[tid] = tid;

   	grid_group grid = this_grid();
   	grid.sync();

	if(tid < num_scc){
		int row_start = trc_row[tid], row_end = trc_row[tid+1];

		for(int i = row_start; i < row_end; i++){
			int v = trc_column[i];

			// With auxilary set data-structure, this linear scan can be reduced to constant time check
			for(int j = trc_row[v]; j < trc_row[v+1]; j++){
				if(trc_column[j] == tid){
					if(colors[tid] > colors[v])
						colors[tid] = colors[v];
					break;
				}
			}
		}
	}

	grid.sync();

	for(int i = 0; i <= m / num_scc; i++){
		int index = i * num_scc + tid;

		if(tid < num_scc && index < m)
			scc[index] = colors[scc[index]];
	}
}

int rename_colors(int m, int* h_scc){
	int *d_colors, *d_scc, *num_scc;
	GPU_ERR_CHK(cudaMalloc(&d_colors, m * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_scc, m * sizeof(int)));
	num_scc = (int *)malloc(sizeof(int));
	GPU_ERR_CHK(cudaMemset(d_colors, 0, m * sizeof(int)))
	GPU_ERR_CHK(cudaMemcpy(d_scc, h_scc, m * sizeof(int), cudaMemcpyHostToDevice));

	int nthreads = MAXBLOCKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	void *kernelArgs[] = {
	   (void*)&m,
	   (void*)&d_scc,
	   (void*)&d_colors,
	};
	dim3 dimBlock(nthreads, 1, 1);
	dim3 dimGrid(nblocks, 1, 1);

	GPU_ERR_CHK(cudaLaunchCooperativeKernel((void*)update_colors, dimGrid, dimBlock, kernelArgs, 0, NULL));
	GPU_ERR_CHK(cudaMemcpy(h_scc, d_scc, m * sizeof(int), cudaMemcpyDeviceToHost));
	GPU_ERR_CHK(cudaMemcpy(num_scc, &(d_scc[m-1]), sizeof(int), cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaFree(d_colors));
	GPU_ERR_CHK(cudaFree(d_scc));

	int ret = (1+*num_scc);
	free(num_scc);
	return ret;
}

__global__ void transitive_closure_kernel(int k, int num_scc, stdgpu::unordered_set<int> edge_set[]){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < num_scc){
		stdgpu::unordered_set<int> new_nodes;
		for (const auto& u: edge_set[tid]) {
			if(edge_set[u].find(k) != edge_set[u].end() && edge_set[tid].find(k) == edge_set[tid].end()){
				new_nodes.insert(k);
			}
		}
		
		edge_set[tid].insert(new_nodes);
	}
}

void update_transitive_closure(const int num_scc, int*& trc_column, int* trc_row){
	int nthreads = MAXBLOCKSIZE;
	int nblocks = (num_scc - 1) / nthreads + 1;
	int nnz = 0;
	std::unordered_set<int> edge_set[num_scc];
	stdgpu::unordered_set<int> d_edge_set[num_scc];

	for(int i = 0; i < num_scc; i++){
		int row_start = trc_row[i], row_end = trc_row[i+1];
		for(int j = row_start; j < row_end; j++){
			edge_set[i].insert(trc_column[j]);
			d_edge_set[i].insert(trc_column[j]);
		}
	}

	for(int k = 0; k <= num_scc; k++){
		transitive_closure_kernel<<<nblocks, nthreads>>>(k, num_scc, d_edge_set);
	}

	for(int i = 0; i < num_scc; i++){
		nnz += d_edge_set[i].size();
	}

	free(trc_column);
	trc_column = (int *)malloc(nnz * sizeof(int));
	trc_row[num_scc] = nnz;
	printf("scc-nnz: %d\n", nnz);

	int col = 0;
	for(int i = 0; i < num_scc; i++){
		trc_row[i] = col;
		for (const auto& u: edge_set[i]) {
			trc_column[col] = u;
			col++;
		}
	}
}

void merge_scc(int m, int num_scc, int* h_scc, int* h_trc_column, int* h_trc_row){
	int nthreads = MAXBLOCKSIZE;
	int nblocks = (num_scc - 1) / nthreads + 1;
	int nnz = h_trc_row[num_scc];
   
	int *d_scc, *d_colors;
	int *d_trc_column, *d_trc_row;
	GPU_ERR_CHK(cudaMalloc(&d_scc, m * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_trc_column, nnz * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_trc_row, (1 + num_scc) * sizeof(int)));
	GPU_ERR_CHK(cudaMalloc(&d_colors, num_scc * sizeof(int)));
	GPU_ERR_CHK(cudaMemcpy(d_scc, h_scc, m * sizeof(int), cudaMemcpyHostToDevice));
	GPU_ERR_CHK(cudaMemcpy(d_trc_column, h_trc_column, nnz * sizeof(int), cudaMemcpyHostToDevice));
	GPU_ERR_CHK(cudaMemcpy(d_trc_row, h_trc_row, (1 + num_scc) * sizeof(int), cudaMemcpyHostToDevice));

	void *kernelArgs[] = {
	   (void*)&m,
	   (void*)&num_scc,
	   (void*)&nnz,
	   (void*)&d_scc,
	   (void*)&d_colors,
	   (void*)&d_trc_column,
	   (void*)&d_trc_row
	};
	dim3 dimBlock(nthreads, 1, 1);
	dim3 dimGrid(nblocks, 1, 1);

	GPU_ERR_CHK(cudaLaunchCooperativeKernel((void*)merge_scc_kernel, dimGrid, dimBlock, kernelArgs, 0, NULL));
	GPU_ERR_CHK(cudaMemcpy(h_scc, d_scc, m * sizeof(int), cudaMemcpyDeviceToHost));

	GPU_ERR_CHK(cudaFree(d_scc));
	GPU_ERR_CHK(cudaFree(d_trc_column));
	GPU_ERR_CHK(cudaFree(d_trc_row));
	GPU_ERR_CHK(cudaFree(d_colors));
}

void initialize_transitive_closure(int* row_offsets, int* column_indices, int num_scc, int* scc_root, int*& trc_column, int* trc_row){

   //int dev = 0;
   //cudaDeviceProp deviceProp;
   //cudaGetDeviceProperties(&deviceProp, dev);
   //printf("Maximum Active Blocks (assuming one block per SM)= %d\n", deviceProp.multiProcessorCount);
   //printf("Maximum size of graph = %d nodes\n", deviceProp.multiProcessorCount*MAXBLOCKSIZE);
   
	int nnz = 0;
	vector<int> adj_list[num_scc];

   	for(int i = 1; i <= num_scc; i++){
	  	int row_start = row_offsets[i-1], row_end = row_offsets[i];
	  	int u = scc_root[i-1], v;
	  
	  	for(int j = row_start; j < row_end; j++){
		 	v = scc_root[column_indices[j]];
		 	adj_list[u].push_back(v);
			nnz++;
	  	}
   	}

	trc_column = (int *)malloc(nnz * sizeof(int));
	trc_row[num_scc] = nnz;

	int col = 0;
   	for(int i = 0; i < num_scc; i++){
		trc_row[i] = col;

		for(int j = 0; j < adj_list[i].size(); j++){
			trc_column[col] = adj_list[i][j];
			col++;
		}
	}
}

void read_updates(char* file_path, int m, int num_scc, int* out_row_offsets, int*& out_column_indices, int* scc_root, int*& trc_column, int* trc_row){
   	FILE* file = fopen(file_path, "r");
		 
   	if(!file){  
	  	printf("Update file can't be read\n"); 
	  	exit(-1); 
   	} 

	int x, y;
	std::unordered_set<int> edge_set[m], trc_edge_set[num_scc];


	// initialise the sets with existing values
	for(int i = 0; i < m; i++){
		x = scc_root[i];

		int row_start = out_row_offsets[i], row_end = out_row_offsets[i+1];
		for(int j = row_start; j < row_end; j++){
			int v = out_column_indices[j];
			edge_set[i].insert(v);

	  		y = scc_root[v];
			trc_edge_set[x].insert(y);
		}
	}


	// read inputs into the sets
   	while(fscanf(file, "%d", &x) == 1 && fscanf(file, "%d", &y) == 1)  
   	{
	  	x--; y--;
	  	if(x >= m || y >= m){
		 	printf("Node %d or %d in update file doesn't exist\n", x, y); 
		 	exit(-1); 
	  	}

	  	edge_set[x].insert(y);

	  	x = scc_root[x];
	  	y = scc_root[y];
	  	trc_edge_set[x].insert(y);
   	}


	// compute number of non-zero values
	int nnz = 0, trc_nnz = 0;

	for(int i = 0; i < m; i++){
		nnz += edge_set[i].size();
	}

	for(int i = 0; i < num_scc; i++){
		trc_nnz += trc_edge_set[i].size();
	}


	// update the graphs from sets
   	free(out_column_indices);
	out_column_indices = (int *)malloc(nnz * sizeof(int));
	out_row_offsets[m] = nnz;

	int col = 0;
	for(int i = 0; i < m; i++){
		out_row_offsets[i] = col;
		for (const auto& u: edge_set[i]) {
			out_column_indices[col] = u;
			col++;
		}
	}

	free(trc_column);
	trc_column = (int *)malloc(trc_nnz * sizeof(int));
	trc_row[num_scc] = trc_nnz;

	col = 0;
	for(int i = 0; i < num_scc; i++){
		trc_row[i] = col;
		for (const auto& u: trc_edge_set[i]) {
			trc_column[col] = u;
			col++;
		}
	}
}