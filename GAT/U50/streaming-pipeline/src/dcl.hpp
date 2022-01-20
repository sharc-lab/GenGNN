#ifndef __DCL_H__
#define __DCL_H__

#include <cstddef>
#include "/tools/reconfig/xilinx/Vitis_HLS/2020.2/include/gmp.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include <algorithm>
#include "hls_math.h"
#include <string.h>
#include <limits.h>

typedef ap_fixed<28, 10> FM_TYPE;
typedef ap_fixed<28, 10> WT_TYPE;

#define MAX_EDGE 500
#define MAX_NODE 200
#define LAYER_NUM 5
#define HEAD_NUM 4  // Max number of head
#define FEATURE_IN 9
#define FEATURE_OUT 16  // Max number of feature
#define NUM_TASK 1

#define ND_FEATURE 9
#define EDGE_ATTR 3

extern int num_heads_per_layer[LAYER_NUM + 1];
extern int num_features_per_layer[LAYER_NUM + 1];


extern float graph_pred_linear_weight[NUM_TASK][FEATURE_OUT];
extern float graph_pred_linear_bias[NUM_TASK];
extern float gat_net_scoring_fn_target[LAYER_NUM][HEAD_NUM][FEATURE_OUT];
extern float gat_net_scoring_fn_source[LAYER_NUM][HEAD_NUM][FEATURE_OUT];
extern float gat_net_0_linear_proj_weight[HEAD_NUM * FEATURE_OUT][FEATURE_IN];
extern float gat_net_1_linear_proj_weight[LAYER_NUM - 1][HEAD_NUM * FEATURE_OUT][HEAD_NUM * FEATURE_OUT];
extern float gat_net_0_skip_proj_weight[HEAD_NUM * FEATURE_OUT][FEATURE_IN];
extern float gat_net_1_skip_proj_weight[LAYER_NUM - 1][HEAD_NUM * FEATURE_OUT][HEAD_NUM * FEATURE_OUT];


extern "C" {
void GAT_compute_one_graph(
    int* node_feature, int* edge_list, int* graph_attr, FM_TYPE* task_tb,
    WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * FEATURE_OUT],
    WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK],
    WT_TYPE gat_net_scoring_fn_target_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT],
    WT_TYPE gat_net_scoring_fn_source_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT],
    WT_TYPE gat_net_linear_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT],
    WT_TYPE gat_net_skip_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT]
    );
}

#endif
