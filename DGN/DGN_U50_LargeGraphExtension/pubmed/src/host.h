#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> embedding_FC_weight_in;
extern aligned_vector<WT_TYPE> embedding_FC_bias_in;
extern aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_weight_in;
extern aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_bias_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_weight_in;
extern aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_bias_in;

void load_weights();
void fetch_one_graph(char* graph_name, aligned_vector<int>& node_feature, aligned_vector<WT_TYPE>& node_eigen, aligned_vector<int>& edge_list, int num_of_nodes, int num_of_edges);
void prepare_graph(int num_of_nodes, int num_of_edges, aligned_vector<int>& edge_list, aligned_vector<int>& degree_table, aligned_vector<int>& neighbor_table);

#endif
