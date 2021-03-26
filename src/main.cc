#include "graph_io.h"
#include "scc.h"
//#include "kernel.h" 		// For dense transitive closure representation
#include "kernel_sparse.h" 	// For sparse transitive closure representation
							// Modify the makefile also to switch between dense and sparse

int main(int argc, char *argv[]) {
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 4) {
		printf("Usage: %s <graph> <edge_updates> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	}
	is_directed = atoi(argv[3]);
	if(!is_directed) symmetrize = true;

	// CSR data structures
	int m, n, nnz;
	WeightT *h_weight = NULL;

	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, n, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, symmetrize, false, false);
	read_graph(argc, argv, m, n, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, symmetrize, true, false);

	int *scc_root = (int *)malloc(m * sizeof(int)), num_scc;
	SCCSolver(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, scc_root);
	printf("SCC calculation of base graph completed!\n\n");

	printf("Running SCC update....\n");
	Timer t_initialize, t_read, t_compute_1, t_compute_2;
	t_compute_1.Start();
	num_scc = rename_colors(m, scc_root);
	t_compute_1.Stop();
	
	
	// Dense representation
	
	/*
	t_initialize.Start();
	bool *trans_closure = (bool *)calloc(num_scc * num_scc, sizeof(bool));
	initialize_transitive_closure(out_row_offsets, out_column_indices, num_scc, scc_root, trans_closure);
	t_initialize.Stop();
	
	t_read.Start();
	read_updates(argv[2], m, num_scc, out_row_offsets, out_column_indices, scc_root, trans_closure);
	t_read.Stop();
	
	float sum = 0;
	t_compute_2.Start();
	update_transitive_closure(num_scc, trans_closure);
	merge_scc(m, num_scc, scc_root, trans_closure);
	num_scc = rename_colors(m, scc_root);
	// TODO: transitive closure must re-initialized for dynamic algorithm
	t_compute_2.Stop();
	*/


	// Sparse representation

	t_initialize.Start();
	int *trc_column = (int *)malloc(nnz * sizeof(int));
	int *trc_row = (int *)malloc((num_scc + 1) * sizeof(int));
	initialize_transitive_closure(out_row_offsets, out_column_indices, num_scc, scc_root, trc_column, trc_row);
	t_initialize.Stop();

	t_read.Start();
	read_updates(argv[2], m, num_scc, out_row_offsets, out_column_indices, scc_root, trc_column, trc_row);
	t_read.Stop();

	t_compute_2.Start();
	update_transitive_closure_cpu(num_scc, trc_column, trc_row);
	merge_scc(m, num_scc, scc_root, trc_column, trc_row);
	num_scc = rename_colors(m, scc_root);
	// TODO: transitive closure must re-initialized for dynamic algorithm
	t_compute_2.Stop();
	

	printf("Completed!\n");
	printf("SCC count = %d\n", num_scc);
	printf("Runtime for initializing Transitive Closure = %f ms.\n", t_initialize.Millisecs());
	printf("Runtime for reading updates from file = %f ms.\n", t_read.Millisecs());
	printf("Runtime for SCC Update = %f ms.\n", t_compute_1.Millisecs() + t_compute_2.Millisecs());
	
	/*
	for(int i=0; i<m; i++){
		printf("%d ", scc_root[i]);
	}
	printf("\n");
	*/

	/*
	for(int i=0; i<num_scc; i++){
		for(int j=0; j<num_scc; j++){
			printf("%d ", trans_closure[i*num_scc + j]);
		}
		printf("\n");
	}
	printf("\n");
	*/


	free(in_row_offsets);
	free(in_column_indices);
	free(in_degree);
	free(out_row_offsets);
	free(out_column_indices);
	free(out_degree);
	free(h_weight);
	free(scc_root);
	//free(trans_closure);
	free(trc_column);
	free(trc_row);
	return 0;
}

