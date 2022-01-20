#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_0_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_1_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_2_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_3_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_4_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_5_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_6_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_7_weight_fixed_in;
extern aligned_vector<WT_TYPE> node_emb_atom_embedding_list_8_weight_fixed_in;

extern aligned_vector<WT_TYPE> mlp_0_weight_fixed_in;
extern aligned_vector<WT_TYPE> mlp_0_bias_fixed_in;
extern aligned_vector<WT_TYPE> mlp_2_weight_fixed_in;
extern aligned_vector<WT_TYPE> mlp_2_bias_fixed_in;
extern aligned_vector<WT_TYPE> mlp_4_weight_fixed_in;
extern aligned_vector<WT_TYPE> mlp_4_bias_fixed_in;

extern aligned_vector<WT_TYPE> convs_ALL_post_nn_0_weight_fixed_in;
extern aligned_vector<WT_TYPE> convs_ALL_post_nn_0_bias_fixed_in;


// void load_weights();
// void fetch_one_graph(char* graph_name, int* node_feature, int* edge_list, int* edge_attr, int num_of_nodes, int num_of_edges);

void load_weights();
void fetch_one_graph(char* graph_name, aligned_vector<int>& node_feature, aligned_vector<int>& edge_list, aligned_vector<int>& edge_attr, int num_of_nodes, int num_of_edges);
#endif