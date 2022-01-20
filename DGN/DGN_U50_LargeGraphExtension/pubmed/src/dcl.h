#ifndef __DCL_H__
#define __DCL_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_int.h>
#include <ap_fixed.h>

// #define GRAPH_CORA
// #define GRAPH_CITESEER
#define GRAPH_PUBMED

#ifdef GRAPH_CORA
#define ND_FEATURE 1433
#endif

#ifdef GRAPH_CITESEER
#define ND_FEATURE 3703
#endif

#ifdef GRAPH_PUBMED
#define ND_FEATURE 500
#endif

#define EMB_DIM 100
#define NUM_TASK 1
#define L_IN 200
#define L_OUT 100

typedef ap_fixed<16, 5> FM_TYPE;
typedef ap_fixed<16, 5> WT_TYPE;
typedef ap_uint<128> INEMB_XFER_TYPE; // transfers for input node embedding
typedef ap_uint<128> XFER_TYPE; // all other internal transfers

#define PACK(packed, index, element) ((packed).range(((index) + 1) * ((element).width) - 1, (index) * ((element).width)) = (element).range((element).width - 1, 0))
#define UNPACK(element, packed, index) ((element).range((element).width - 1, 0) = (packed).range(((index) + 1) * ((element).width) - 1, (index) * ((element).width)))
#define CEILDIV(a, b) (((a) + (b) - 1) / (b))

#define XFER_PER_EMB (CEILDIV(EMB_DIM * FM_TYPE::width, XFER_TYPE::width))
#define INEMB_XFER_PER_ND_FEATURE (CEILDIV(ND_FEATURE * sizeof(int) * 8, INEMB_XFER_TYPE::width))
#define FM_PER_XFER (XFER_TYPE::width / FM_TYPE::width)
#define INT_PER_INEMB_XFER (INEMB_XFER_TYPE::width / (sizeof(int) * 8))

// Options affecting DSP utilization
#define MAX_MUL_CYCLES_PER_NODE 50
#define DIMS_PER_MUL_CYCLE(dim_in) (CEILDIV(dim_in, MAX_MUL_CYCLES_PER_NODE))
#define MUL_CYCLES_PER_NODE(dim_in) (CEILDIV(dim_in, DIMS_PER_MUL_CYCLE(dim_in)))

extern "C" {
void DGN_compute_one_graph(
    float* out,
    int* node_feature_in,
    WT_TYPE* node_eigen_in,
    int degree_table[][2],
    int neighbor_table[],
    int* graph_attr,
    WT_TYPE embedding_FC_weights_in[EMB_DIM][ND_FEATURE],
    WT_TYPE embedding_FC_bias_in[EMB_DIM],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[4][100][2][100],
    WT_TYPE layers_posttrans_fully_connected_0_linear_bias_in[4][100],
    WT_TYPE MLP_layer_FC_layers_0_weight_in[50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias_in[50],
    WT_TYPE MLP_layer_FC_layers_1_weight_in[25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias_in[25],
    WT_TYPE MLP_layer_FC_layers_2_weight_in[1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias_in[1],

    // DRAM for intermediate storage
    FM_TYPE h_node_ping[][EMB_DIM],
    FM_TYPE h_node_pong[][EMB_DIM]
);
}

#endif
