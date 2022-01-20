#include <stdlib.h>
#include <stdio.h>
#include "dcl.hpp"
#include "xcl2.hpp"

int nd_feature_table[ND_FEATURE] = {119, 4, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};


float node_embedding_weight_float[ND_FEATURE_TOTAL][EMB_DIM];
float edge_embedding_weight_float[EG_FEATURE_TOTAL][EMB_DIM];

float node_embedding_weight_raw[ND_FEATURE_TOTAL * EMB_DIM];
float edge_embedding_weight_raw[LAYER_NUM][EG_FEATURE_PER_LAYER * EMB_DIM];

float convs_weight_float[LAYER_NUM][MLP_0_OUT][MLP_0_IN];
float convs_bias_float[LAYER_NUM][MLP_0_OUT];
float convs_root_emb_weight_float[LAYER_NUM][MLP_0_OUT];

float bn_weight_float[LAYER_NUM][EMB_DIM];
float bn_bias_float[LAYER_NUM][EMB_DIM];
float bn_mean_float[LAYER_NUM][EMB_DIM];
float bn_var_float[LAYER_NUM][EMB_DIM];

float graph_pred_linear_weight_float[NUM_TASK][EMB_DIM];
float graph_pred_linear_bias_float[NUM_TASK];


// WT_TYPE node_embedding_weight_fixed[ND_FEATURE_TOTAL][EMB_DIM];
// WT_TYPE edge_embedding_weight_fixed[EG_FEATURE_TOTAL][EMB_DIM];

// WT_TYPE convs_weight_fixed[LAYER_NUM][MLP_0_OUT][MLP_0_IN];
// WT_TYPE convs_bias_fixed[LAYER_NUM][MLP_0_OUT];
// WT_TYPE convs_root_emb_weight_fixed[LAYER_NUM][MLP_0_OUT];

// WT_TYPE bn_weight_fixed[LAYER_NUM][EMB_DIM];
// WT_TYPE bn_bias_fixed[LAYER_NUM][EMB_DIM];
// WT_TYPE bn_mean_fixed[LAYER_NUM][EMB_DIM];
// WT_TYPE bn_var_fixed[LAYER_NUM][EMB_DIM];

// WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK][EMB_DIM];
// WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK];


extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> node_embedding_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> edge_embedding_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_bias_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_root_emb_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_bias_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_mean_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_var_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed;


void load_weights()
{
	printf("Loading weights for GCN ...\n");

    FILE* f;
    f = fopen("./gcn_ep1_dim100.weights.all.bin", "rb");


	fseek(f, 0*sizeof(float), SEEK_SET);
	fread(node_embedding_weight_raw, sizeof(float), ND_FEATURE_TOTAL * EMB_DIM, f);

	fseek(f, 17300*sizeof(float), SEEK_SET);
	fread(convs_weight_float[0], sizeof(float), 10000, f);

	fseek(f, 27300*sizeof(float), SEEK_SET);
	fread(convs_bias_float[0], sizeof(float), 100, f);

	fseek(f, 27400*sizeof(float), SEEK_SET);
	fread(convs_root_emb_weight_float[0], sizeof(float), 100, f);

	fseek(f, 27500*sizeof(float), SEEK_SET);
	fread(edge_embedding_weight_raw[0], sizeof(float), EG_FEATURE_PER_LAYER * EMB_DIM, f);


	fseek(f, 28800*sizeof(float), SEEK_SET);
	fread(convs_weight_float[1], sizeof(float), 10000, f);

	fseek(f, 38800*sizeof(float), SEEK_SET);
	fread(convs_bias_float[1], sizeof(float), 100, f);

	fseek(f, 38900*sizeof(float), SEEK_SET);
	fread(convs_root_emb_weight_float[1], sizeof(float), 100, f);

	fseek(f, 39000*sizeof(float), SEEK_SET);
	fread(edge_embedding_weight_raw[1], sizeof(float), EG_FEATURE_PER_LAYER * EMB_DIM, f);


	fseek(f, 40300*sizeof(float), SEEK_SET);
	fread(convs_weight_float[2], sizeof(float), 10000, f);

	fseek(f, 50300*sizeof(float), SEEK_SET);
	fread(convs_bias_float[2], sizeof(float), 100, f);

	fseek(f, 50400*sizeof(float), SEEK_SET);
	fread(convs_root_emb_weight_float[2], sizeof(float), 100, f);

	fseek(f, 50500*sizeof(float), SEEK_SET);
	fread(edge_embedding_weight_raw[2], sizeof(float), EG_FEATURE_PER_LAYER * EMB_DIM, f);

	fseek(f, 51800*sizeof(float), SEEK_SET);
	fread(convs_weight_float[3], sizeof(float), 10000, f);

	fseek(f, 61800*sizeof(float), SEEK_SET);
	fread(convs_bias_float[3], sizeof(float), 100, f);

	fseek(f, 61900*sizeof(float), SEEK_SET);
	fread(convs_root_emb_weight_float[3], sizeof(float), 100, f);

	fseek(f, 62000*sizeof(float), SEEK_SET);
	fread(edge_embedding_weight_raw[3], sizeof(float), EG_FEATURE_PER_LAYER * EMB_DIM, f);


	fseek(f, 63300*sizeof(float), SEEK_SET);
	fread(convs_weight_float[4], sizeof(float), 10000, f);

	fseek(f, 73300*sizeof(float), SEEK_SET);
	fread(convs_bias_float[4], sizeof(float), 100, f);

	fseek(f, 73400*sizeof(float), SEEK_SET);
	fread(convs_root_emb_weight_float[4], sizeof(float), 100, f);

	fseek(f, 73500*sizeof(float), SEEK_SET);
	fread(edge_embedding_weight_raw[4], sizeof(float), EG_FEATURE_PER_LAYER * EMB_DIM, f);

	
	fseek(f, 74800*sizeof(float), SEEK_SET);
	fread(bn_weight_float[0], sizeof(float), 100, f);

	fseek(f, 74900*sizeof(float), SEEK_SET);
	fread(bn_bias_float[0], sizeof(float), 100, f);

	fseek(f, 75000*sizeof(float), SEEK_SET);
	fread(bn_mean_float[0], sizeof(float), 100, f);

	fseek(f, 75100*sizeof(float), SEEK_SET);
	fread(bn_var_float[0], sizeof(float), 100, f);


	fseek(f, 75201*sizeof(float), SEEK_SET);
	fread(bn_weight_float[1], sizeof(float), 100, f);

	fseek(f, 75301*sizeof(float), SEEK_SET);
	fread(bn_bias_float[1], sizeof(float), 100, f);

	fseek(f, 75401*sizeof(float), SEEK_SET);
	fread(bn_mean_float[1], sizeof(float), 100, f);

	fseek(f, 75501*sizeof(float), SEEK_SET);
	fread(bn_var_float[1], sizeof(float), 100, f);

		
	fseek(f, 75602*sizeof(float), SEEK_SET);
	fread(bn_weight_float[2], sizeof(float), 100, f);

	fseek(f, 75702*sizeof(float), SEEK_SET);
	fread(bn_bias_float[2], sizeof(float), 100, f);

	fseek(f, 75802*sizeof(float), SEEK_SET);
	fread(bn_mean_float[2], sizeof(float), 100, f);

	fseek(f, 75902*sizeof(float), SEEK_SET);
	fread(bn_var_float[2], sizeof(float), 100, f);


	fseek(f, 76003*sizeof(float), SEEK_SET);
	fread(bn_weight_float[3], sizeof(float), 100, f);

	fseek(f, 76103*sizeof(float), SEEK_SET);
	fread(bn_bias_float[3], sizeof(float), 100, f);

	fseek(f, 76203*sizeof(float), SEEK_SET);
	fread(bn_mean_float[3], sizeof(float), 100, f);

	fseek(f, 76303*sizeof(float), SEEK_SET);
	fread(bn_var_float[3], sizeof(float), 100, f);
	

	fseek(f, 76404*sizeof(float), SEEK_SET);
	fread(bn_weight_float[4], sizeof(float), 100, f);

	fseek(f, 76504*sizeof(float), SEEK_SET);
	fread(bn_bias_float[4], sizeof(float), 100, f);

	fseek(f, 76604*sizeof(float), SEEK_SET);
	fread(bn_mean_float[4], sizeof(float), 100, f);

	fseek(f, 76704*sizeof(float), SEEK_SET);
	fread(bn_var_float[4], sizeof(float), 100, f);
	

	fseek(f, 76805*sizeof(float), SEEK_SET);
	fread(graph_pred_linear_weight_float, sizeof(float), 100, f);

	fseek(f, 76905*sizeof(float), SEEK_SET);
	fread(graph_pred_linear_bias_float, sizeof(float), 1, f);

	fclose(f);


	int idx = 0;
	for(int i = 0; i < ND_FEATURE; i++) {
		int nd_f = nd_feature_table[i];
		for(int j = 0; j < nd_f; j++) {
			for(int dim = 0; dim < EMB_DIM; dim++) {
				node_embedding_weight_float[idx + j][dim] = node_embedding_weight_raw[(idx + j) * EMB_DIM + dim];
			}
		}
		idx += nd_f;
	}

	idx = 0;
	for(int l = 0; l < LAYER_NUM; l++) {
		int	idx2 = 0;
		for(int i = 0; i < EDGE_ATTR; i++) {
			int ed_f = ed_feature_table[i];
			for(int j = 0; j < ed_f; j++) {
				for(int dim = 0; dim < EMB_DIM; dim++) {
					edge_embedding_weight_float[idx + j][dim] = edge_embedding_weight_raw[l][(idx2 + j) * EMB_DIM + dim];
				}
			}
			idx += ed_f;
			idx2 += ed_f;
		}
	}


	//////// Convert to fixed point
	for (int i = 0; i < ND_FEATURE_TOTAL; i++) {
		for(int j = 0; j < EMB_DIM; j++) {
			node_embedding_weight_fixed[i * EMB_DIM + j] = (WT_TYPE)node_embedding_weight_float[i][j];
		}
	}
	
	for (int i = 0; i < EG_FEATURE_TOTAL; i++) {
		for(int j = 0; j < EMB_DIM; j++) {
			edge_embedding_weight_fixed[i * EMB_DIM + j] = (WT_TYPE)edge_embedding_weight_float[i][j];
		}
	}

	for(int i = 0; i < LAYER_NUM; i++) {
		for(int j = 0; j < MLP_0_OUT; j++) {
			convs_bias_fixed[i * MLP_0_OUT + j] = (WT_TYPE)convs_bias_float[i][j];
			convs_root_emb_weight_fixed[i * MLP_0_OUT + j] = (WT_TYPE)convs_root_emb_weight_float[i][j];

			for(int k = 0; k < MLP_0_IN; k++) {
				convs_weight_fixed[i * MLP_0_OUT * MLP_0_IN + j * MLP_0_IN + k] = (WT_TYPE)convs_weight_float[i][j][k];
			}
		}
	}

	for(int i = 0; i < LAYER_NUM; i++) {
		for(int j = 0; j < EMB_DIM; j++) {
			bn_weight_fixed[i * EMB_DIM + j] = (WT_TYPE)bn_weight_float[i][j];
			bn_bias_fixed[i * EMB_DIM + j] = (WT_TYPE)bn_bias_float[i][j];
			bn_mean_fixed[i * EMB_DIM + j] = (WT_TYPE)bn_mean_float[i][j];
			bn_var_fixed[i * EMB_DIM + j] = (WT_TYPE)bn_var_float[i][j];
		}
	}
		
	for(int i = 0; i < NUM_TASK; i++) {
		graph_pred_linear_bias_fixed[i] = (WT_TYPE)graph_pred_linear_bias_float[i];
		for(int j = 0; j < EMB_DIM; j++) {
			graph_pred_linear_weight_fixed[i * EMB_DIM + j] = (WT_TYPE)graph_pred_linear_weight_float[i][j];
		}
	}



}



void fetch_one_graph(char* graph_name, std::vector<int, aligned_allocator<int>>* node_feature, std::vector<int, aligned_allocator<int>>* edge_list, std::vector<int, aligned_allocator<int>>* edge_attr, int num_of_nodes, int num_of_edges)
{
    printf("Loading graph ...\n");
        
    FILE* f;

	char f_node_feature[128];
	char f_edge_list[128];
	char f_edge_attr[128];

	sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
	sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
	sprintf(f_edge_attr, "%s_edge_attr.bin", graph_name);
	
	int node_feature_in[ND_FEATURE * MAX_NODE];
	int edge_list_in[2 * MAX_EDGE];
    int edge_attr_in[EDGE_ATTR * MAX_EDGE];
        

    f = fopen(f_node_feature, "r");
	fread(node_feature_in, sizeof(int), num_of_nodes * ND_FEATURE, f);
    fclose(f);

    f = fopen(f_edge_list, "r");
    fread(edge_list_in, sizeof(int), 2 * num_of_edges, f);
    fclose(f);

    f = fopen(f_edge_attr, "r");
    fread(edge_attr_in, sizeof(int), EDGE_ATTR * num_of_edges, f);
    fclose(f);

	for(int i = 0; i < num_of_nodes * ND_FEATURE; i++) {
		(*node_feature)[i] = node_feature_in[i];
	}

	for(int i = 0; i < 2 * num_of_edges; i++) {
		(*edge_list)[i] = edge_list_in[i];
	}

	for(int i = 0; i < num_of_edges * EDGE_ATTR; i++) {
		(*edge_attr)[i] = edge_attr_in[i];
	}


}
