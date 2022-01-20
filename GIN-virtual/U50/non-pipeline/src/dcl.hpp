#ifndef __DCL_H__
#define __DCL_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cstddef>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include <algorithm>


typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;


/////////// Model Specific Configurations /////////////
#define MAX_EDGE 500
#define MAX_NODE 200
#define MAX_DEGREE 20

#define LAYER_NUM 5
#define EMB_DIM 100
#define NUM_TASK 1
#define MLP_1_IN 100
#define MLP_1_OUT 200
#define MLP_2_IN 200
#define MLP_2_OUT 100
#define MLP_IN_MAX 200
#define MLP_OUT_MAX 200
#define E_EPS 0.00001

#define ND_FEATURE 9
#define EDGE_ATTR 3

extern int nd_feature_table[ND_FEATURE]; // = {119, 4, 12, 12, 10, 6, 6, 2};
#define ND_FEATURE_TOTAL 173 // 119 + 4 + ... + 2
extern int ed_feature_table[EDGE_ATTR]; // = {5, 6, 2};
#define EG_FEATURE_TOTAL 65 // (5 + 6 + 2) * LAYER_NUM

/////////// Model weights /////////////

extern float gnn_node_mlp_1_weights[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
extern float gnn_node_mlp_1_bias[LAYER_NUM][MLP_1_OUT];
extern float gnn_node_mlp_2_weights[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
extern float gnn_node_mlp_2_bias[LAYER_NUM][MLP_2_OUT];
extern float gnn_node_embedding_table[ND_FEATURE_TOTAL][EMB_DIM];
extern float gnn_edge_embedding_table[EG_FEATURE_TOTAL][EMB_DIM];
extern float graph_pred_linear_weight[NUM_TASK][MLP_2_OUT];
extern float graph_pred_linear_bias[NUM_TASK];
extern float eps[LAYER_NUM];


// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_1_weights_fixed(LAYER_NUM * MLP_1_OUT * MLP_1_IN);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_1_bias_fixed(LAYER_NUM * MLP_1_OUT);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_2_weights_fixed(LAYER_NUM * MLP_2_OUT * MLP_2_IN);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_2_bias_fixed(LAYER_NUM * MLP_2_OUT);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_embedding_table_fixed(ND_FEATURE_TOTAL * EMB_DIM);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_edge_embedding_table_fixed(EG_FEATURE_TOTAL * EMB_DIM);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed(NUM_TASK * MLP_2_OUT);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed(NUM_TASK);
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> eps_fixed(LAYER_NUM);


extern "C" {
void GIN_compute_one_graph(
    int* node_feature_in, int* edge_list_in, int* edge_attr_in, int* graph_attr, FM_TYPE* task,
    WT_TYPE gnn_node_mlp_1_weights_fixed[LAYER_NUM * MLP_1_OUT * MLP_1_IN],
    WT_TYPE gnn_node_mlp_1_bias_fixed[LAYER_NUM * MLP_1_OUT],
    WT_TYPE gnn_node_mlp_2_weights_fixed[LAYER_NUM * MLP_2_OUT * MLP_2_IN],
    WT_TYPE gnn_node_mlp_2_bias_fixed[LAYER_NUM * MLP_2_OUT],
    WT_TYPE gnn_node_embedding_fixed[ND_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE gnn_edge_embedding_fixed[EG_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * MLP_2_OUT],
    WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK],
    WT_TYPE eps_fixed[LAYER_NUM]
    );
}

#endif
