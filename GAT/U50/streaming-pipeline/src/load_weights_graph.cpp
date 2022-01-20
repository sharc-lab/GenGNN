#include <stdlib.h>
#include <stdio.h>
#include "dcl.hpp"
#include "xcl2.hpp"

// extern WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * FEATURE_OUT];
// extern WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK];
// extern WT_TYPE gat_net_scoring_fn_target_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT];
// extern WT_TYPE gat_net_scoring_fn_source_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT];
// extern WT_TYPE gat_net_linear_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT];
// extern WT_TYPE gat_net_skip_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT];
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_scoring_fn_target_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_scoring_fn_source_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_linear_proj_weight_fixed;
extern std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_skip_proj_weight_fixed;

void load_weights()
{
	printf("Loading weights for GAT ...\n");

    FILE* f;

	f = fopen("gat_ep1_pred_weights_layer5.bin", "r");
	fread(graph_pred_linear_weight, sizeof(float), NUM_TASK * FEATURE_OUT, f);
	fclose(f);

	f = fopen("gat_ep1_pred_bias_layer5.bin", "r");
	fread(graph_pred_linear_bias, sizeof(float), NUM_TASK, f);
	fclose(f);

	f = fopen("gat_ep1_scoring_fn_target_layer5.bin", "r");
	fread(gat_net_scoring_fn_target, sizeof(float), LAYER_NUM * HEAD_NUM * FEATURE_OUT, f);
	fclose(f);

	f = fopen("gat_ep1_scoring_fn_source_layer5.bin", "r");
	fread(gat_net_scoring_fn_source, sizeof(float), LAYER_NUM * HEAD_NUM * FEATURE_OUT, f);
	fclose(f);

	f = fopen("gat_ep1_linear_proj_weight_0_layer5.bin", "r");
	fread(gat_net_0_linear_proj_weight, sizeof(float), HEAD_NUM * FEATURE_OUT * FEATURE_IN, f);
	fclose(f);

	f = fopen("gat_ep1_linear_proj_weight_1_layer5.bin", "r");
	fread(gat_net_1_linear_proj_weight, sizeof(float), (LAYER_NUM - 1) * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT, f);
	fclose(f);

	f = fopen("gat_ep1_skip_proj_weight_0_layer5.bin", "r");
	fread(gat_net_0_skip_proj_weight, sizeof(float), HEAD_NUM * FEATURE_OUT * FEATURE_IN, f);
	fclose(f);

	f = fopen("gat_ep1_skip_proj_weight_1_layer5.bin", "r");
	fread(gat_net_1_skip_proj_weight, sizeof(float), (LAYER_NUM - 1) * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT, f);
	fclose(f);

	/// convert to fixed point
	for(int i = 0; i < NUM_TASK; i++) {
		graph_pred_linear_bias_fixed[i] = (WT_TYPE)graph_pred_linear_bias[i];
		for(int j = 0; j < FEATURE_OUT; j++) {
			graph_pred_linear_weight_fixed[i * FEATURE_OUT + j] = (WT_TYPE)graph_pred_linear_weight[i][j];
		}
	}

	for(int i = 0; i < LAYER_NUM; i++) {
		for(int j = 0; j < HEAD_NUM; j++) {
			for (int k = 0; k < FEATURE_OUT; k++) {
				gat_net_scoring_fn_target_fixed[i * HEAD_NUM * FEATURE_OUT + j * FEATURE_OUT + k] = (WT_TYPE)gat_net_scoring_fn_target[i][j][k];
				gat_net_scoring_fn_source_fixed[i * HEAD_NUM * FEATURE_OUT + j * FEATURE_OUT + k] = (WT_TYPE)gat_net_scoring_fn_source[i][j][k];
			}
		}
	}

	for(int i = 0; i < HEAD_NUM * FEATURE_OUT; i++) {
		for (int j = 0; j < FEATURE_IN; j++) {
			gat_net_linear_proj_weight_fixed[i * FEATURE_IN + j] = (WT_TYPE)gat_net_0_linear_proj_weight[i][j];
			gat_net_skip_proj_weight_fixed[i * FEATURE_IN + j] = (WT_TYPE)gat_net_0_skip_proj_weight[i][j];
		}
	}

	for(int i = 1; i < LAYER_NUM; i++) {
		for(int j = 0; j < HEAD_NUM * FEATURE_OUT; j++) {
			for (int k = 0; k < HEAD_NUM * FEATURE_OUT; k++) {
				gat_net_linear_proj_weight_fixed[i * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT + j * HEAD_NUM * FEATURE_OUT + k] = (WT_TYPE)gat_net_1_linear_proj_weight[i - 1][j][k];
				gat_net_skip_proj_weight_fixed[i * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT + j * HEAD_NUM * FEATURE_OUT + k] = (WT_TYPE)gat_net_1_skip_proj_weight[i - 1][j][k];
			}
		}
	}

}

	
void fetch_one_graph(char* graph_name, std::vector<int, aligned_allocator<int>>* node_feature, std::vector<int, aligned_allocator<int>>* edge_list, int num_of_nodes, int num_of_edges)
{
    printf("Loading graph ...\n");
        
    FILE* f;

	char f_node_feature[128];
	char f_edge_list[128];

	sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
	sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
	
	int node_feature_in[ND_FEATURE * MAX_NODE];
	int edge_list_in[2 * MAX_EDGE];
    
    f = fopen(f_node_feature, "r");
	fread(node_feature_in, sizeof(int), num_of_nodes * ND_FEATURE, f);
    fclose(f);

    f = fopen(f_edge_list, "r");
    fread(edge_list_in, sizeof(int), 2 * num_of_edges, f);
    fclose(f);
    
    for(int i = 0; i < num_of_nodes * ND_FEATURE; i++) {
  		(*node_feature)[i] = node_feature_in[i];
  	}
  
  	for(int i = 0; i < 2 * num_of_edges; i++) {
  		(*edge_list)[i] = edge_list_in[i];
  	}

#ifdef _PRINT_
		printf("Node features:\n");
		for(int i = 0; i < num_of_nodes; i++) {
			for(int j = 0; j < ND_FEATURE; j++) {
				printf("%d ", node_feature[i * ND_FEATURE + j]);
			}
			printf("\n");
		}

		printf("Edges:\n");
		for(int i = 0; i < num_of_edges; i++) {
			printf("%d -> %d\n", edge_list[i*2], edge_list[i*2+1]);
		}

		printf("Edge attributes:\n");
		for(int i = 0; i < num_of_edges; i++) {
			for(int j = 0; j < EDGE_ATTR; j++) {
				printf("%d ", edge_attr[i * EDGE_ATTR + j]);
			}
			printf("\n");
		}
	}
#endif
}
