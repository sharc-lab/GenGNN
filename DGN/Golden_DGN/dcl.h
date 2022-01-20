#ifndef __DCL_H__
#define __DCL_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_EDGE 500
#define MAX_NODE 500
#define ND_FEATURE 9
#define EDGE_ATTR 3
#define EMB_DIM 100
#define NUM_TASK 1
#define L_IN 200
#define L_OUT 100

extern float embedding_h_atom_embedding_list_0_weight[119][100];
extern float embedding_h_atom_embedding_list_1_weight[4][100];
extern float embedding_h_atom_embedding_list_2_weight[12][100];
extern float embedding_h_atom_embedding_list_3_weight[12][100];
extern float embedding_h_atom_embedding_list_4_weight[10][100];
extern float embedding_h_atom_embedding_list_5_weight[6][100];
extern float embedding_h_atom_embedding_list_6_weight[6][100];
extern float embedding_h_atom_embedding_list_7_weight[2][100];
extern float embedding_h_atom_embedding_list_8_weight[2][100];
extern float layers_0_posttrans_fully_connected_0_linear_weight[100][200];
extern float layers_0_posttrans_fully_connected_0_linear_bias[100];
extern float layers_1_posttrans_fully_connected_0_linear_weight[100][200];
extern float layers_1_posttrans_fully_connected_0_linear_bias[100];
extern float layers_2_posttrans_fully_connected_0_linear_weight[100][200];
extern float layers_2_posttrans_fully_connected_0_linear_bias[100];
extern float layers_3_posttrans_fully_connected_0_linear_weight[100][200];
extern float layers_3_posttrans_fully_connected_0_linear_bias[100];
extern float MLP_layer_FC_layers_0_weight[50][100];
extern float MLP_layer_FC_layers_0_bias[50];
extern float MLP_layer_FC_layers_1_weight[25][50];
extern float MLP_layer_FC_layers_1_bias[25];
extern float MLP_layer_FC_layers_2_weight[1][25];
extern float MLP_layer_FC_layers_2_bias[1];


void load_weights();
void fetch_one_graph(char* graph_name, int* node_feature, int* edge_list, int* edge_attr, int num_of_nodes, int num_of_edges);
void DGN_compute_one_graph(int g, int* node_feature, int* edge_list, int* edge_attr, int* graph_attr);
bool Jacob(float *pMatrix, int nDim, float *pdblVects, float *pdbEigenValues, float dbEps, int nJt);

#endif
