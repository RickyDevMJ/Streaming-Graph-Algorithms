#include "common.h"
#include <vector>
#include <unordered_set>

int rename_colors(int m, int* h_scc);
void update_transitive_closure(int num_scc, int* h_trc_column, int* h_trc_row);
void update_transitive_closure_cpu(const int num_scc, int*& trc_column, int* trc_row);
void merge_scc(int m, int num_scc, int* h_scc, int* h_trc_column, int* h_trc_row);
void initialize_transitive_closure(int* row_offsets, int* column_indices, int num_scc, int* scc_root, int*& trc_column, int* trc_row);
void read_updates(char* file_path, int m, int num_scc, int* out_row_offsets, int*& out_column_indices, int* scc_root, int*& trc_column, int* tc_row);