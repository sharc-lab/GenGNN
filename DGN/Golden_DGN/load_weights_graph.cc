#include <stdlib.h>
#include <stdio.h>
#include "dcl.h"


void load_weights()
{
	printf("Loading weights for DGN ...\n");

    FILE* f;
    f = fopen("dgn_ep1_noBN_dim100.weights.all.bin", "rb");
	fseek(f, 0*sizeof(float), SEEK_SET);	fseek(f, 0*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_0_weight, sizeof(float), 11900, f);

	fseek(f, 11900*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_1_weight, sizeof(float), 400, f);

	fseek(f, 12300*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_2_weight, sizeof(float), 1200, f);

	fseek(f, 13500*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_3_weight, sizeof(float), 1200, f);

	fseek(f, 14700*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_4_weight, sizeof(float), 1000, f);

	fseek(f, 15700*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_5_weight, sizeof(float), 600, f);

	fseek(f, 16300*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_6_weight, sizeof(float), 600, f);

	fseek(f, 16900*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_7_weight, sizeof(float), 200, f);

	fseek(f, 17100*sizeof(float), SEEK_SET);
	fread(embedding_h_atom_embedding_list_8_weight, sizeof(float), 200, f);

	fseek(f, 17300*sizeof(float), SEEK_SET);
	fread(layers_0_posttrans_fully_connected_0_linear_weight, sizeof(float), 20000, f);

	fseek(f, 37300*sizeof(float), SEEK_SET);
	fread(layers_0_posttrans_fully_connected_0_linear_bias, sizeof(float), 100, f);

	
	fseek(f, 37400*sizeof(float), SEEK_SET);
	fread(layers_1_posttrans_fully_connected_0_linear_weight, sizeof(float), 20000, f);

	fseek(f, 57400*sizeof(float), SEEK_SET);
	fread(layers_1_posttrans_fully_connected_0_linear_bias, sizeof(float), 100, f);
	
	fseek(f, 57500*sizeof(float), SEEK_SET);
	fread(layers_2_posttrans_fully_connected_0_linear_weight, sizeof(float), 20000, f);

	fseek(f, 77500*sizeof(float), SEEK_SET);
	fread(layers_2_posttrans_fully_connected_0_linear_bias, sizeof(float), 100, f);
	
	fseek(f, 77600*sizeof(float), SEEK_SET);
	fread(layers_3_posttrans_fully_connected_0_linear_weight, sizeof(float), 20000, f);

	fseek(f, 97600*sizeof(float), SEEK_SET);
	fread(layers_3_posttrans_fully_connected_0_linear_bias, sizeof(float), 100, f);

	fseek(f, 97700*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_0_weight, sizeof(float), 5000, f);

	fseek(f, 102700*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_0_bias, sizeof(float), 50, f);
	
	fseek(f, 102750*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_1_weight, sizeof(float), 1250, f);

	fseek(f, 104000*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_1_bias, sizeof(float), 25, f);

	fseek(f, 104025*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_2_weight, sizeof(float), 25, f);

	fseek(f, 104050*sizeof(float), SEEK_SET);
	fread(MLP_layer_FC_layers_2_bias, sizeof(float), 1, f);

	fclose(f);

}
void fetch_one_graph(char* graph_name, int* node_feature, int* edge_list, int* edge_attr, int num_of_nodes, int num_of_edges)
{
    printf("Loading graph ...\n");
        
    FILE* f;

	char f_node_feature[128];
	char f_edge_list[128];
	char f_edge_attr[128];

	sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
	sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
	sprintf(f_edge_attr, "%s_edge_attr.bin", graph_name);
	
	
    f = fopen(f_node_feature, "rb");
	fread(node_feature, sizeof(int), num_of_nodes * ND_FEATURE, f);
    fclose(f);


    f = fopen(f_edge_list, "rb");
    fread(edge_list, sizeof(int), 2 * num_of_edges, f);
    fclose(f);


    f = fopen(f_edge_attr, "rb");
    fread(edge_attr, sizeof(int), EDGE_ATTR * num_of_edges, f);
    fclose(f);


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