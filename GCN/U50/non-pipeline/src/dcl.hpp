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
#include "hls_math.h"

#include <algorithm>


typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;

#define MAX_EDGE 500
#define MAX_NODE 200

#define LAYER_NUM 5

#define ND_FEATURE 9
#define EDGE_ATTR 3
#define EMB_DIM 100
#define NUM_TASK 1

#define MLP_0_IN 100
#define MLP_0_OUT 100
#define MLP_IN_MAX 100
#define MLP_OUT_MAX 100
#define E_EPS 0.00001

#define ND_FEATURE 9
#define EDGE_ATTR 3

extern int nd_feature_table[ND_FEATURE]; // = {119, 4, 12, 12, 10, 6, 6, 2, 2};
#define ND_FEATURE_TOTAL 173 // 119 + 4 + ... + 2
extern int ed_feature_table[EDGE_ATTR]; // = {5, 6, 2};
#define EG_FEATURE_PER_LAYER 13 // (5 + 6 + 2)
#define EG_FEATURE_TOTAL 65 // (5 + 6 + 2) * LAYER_NUM



// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> node_embedding_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> edge_embedding_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_bias_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_root_emb_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_bias_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_mean_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_var_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed;
// extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed;



//void load_weights();
//void fetch_one_graph(char* graph_name, int* node_feature, int* edge_list, int* edge_attr, int num_of_nodes, int num_of_edges);

extern "C" {
void GCN_compute_one_graph(
    int* node_feature_in, int* edge_list_in, int* edge_attr_in, int* graph_attr, WT_TYPE* task,
    WT_TYPE convs_weight_in[LAYER_NUM][100][100], WT_TYPE convs_bias_in[LAYER_NUM][100], WT_TYPE convs_root_emb_weight_in[LAYER_NUM][100],
    WT_TYPE bn_weigh_in[LAYER_NUM][100], WT_TYPE bn_bias_in[LAYER_NUM][100], WT_TYPE bn_mean_in[LAYER_NUM][100], WT_TYPE bn_var_in[LAYER_NUM][100],
    WT_TYPE node_embedding_weight_in[ND_FEATURE_TOTAL][EMB_DIM], WT_TYPE edge_embedding_weight_in[EG_FEATURE_TOTAL][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][MLP_0_OUT], WT_TYPE graph_pred_bias_in[NUM_TASK]
    );
}

#endif
