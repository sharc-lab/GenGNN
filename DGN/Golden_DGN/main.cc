#include <stdio.h>
#include <stdlib.h>
#include "dcl.h"
float embedding_h_atom_embedding_list_0_weight[119][100];
float embedding_h_atom_embedding_list_1_weight[4][100];
float embedding_h_atom_embedding_list_2_weight[12][100];
float embedding_h_atom_embedding_list_3_weight[12][100];
float embedding_h_atom_embedding_list_4_weight[10][100];
float embedding_h_atom_embedding_list_5_weight[6][100];
float embedding_h_atom_embedding_list_6_weight[6][100];
float embedding_h_atom_embedding_list_7_weight[2][100];
float embedding_h_atom_embedding_list_8_weight[2][100];
float layers_0_posttrans_fully_connected_0_linear_weight[100][200];
float layers_0_posttrans_fully_connected_0_linear_bias[100];
float layers_1_posttrans_fully_connected_0_linear_weight[100][200];
float layers_1_posttrans_fully_connected_0_linear_bias[100];
float layers_2_posttrans_fully_connected_0_linear_weight[100][200];
float layers_2_posttrans_fully_connected_0_linear_bias[100];
float layers_3_posttrans_fully_connected_0_linear_weight[100][200];
float layers_3_posttrans_fully_connected_0_linear_bias[100];
float MLP_layer_FC_layers_0_weight[50][100];
float MLP_layer_FC_layers_0_bias[50];
float MLP_layer_FC_layers_1_weight[25][50];
float MLP_layer_FC_layers_1_bias[25];
float MLP_layer_FC_layers_2_weight[1][25];
float MLP_layer_FC_layers_2_bias[1];
// global weights
extern float final;

int main()
{
    printf("\n******* This is the golden C file for DGN model *******\n");

    load_weights();

    float all_results[4113];
    FILE* c_output = fopen("Golden_C_output.txt", "w+");
    for(int g = 1; g <= 4113; g++ ) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;


        sprintf(info_file, "../../graphs/graph_info/g%d_info.txt", g);
        sprintf(graph_name, "../../graphs/graph_bin/g%d", g);


        FILE* f_info = fopen(info_file, "r");
        fscanf (f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);
        

        printf("********** Computing Graph %s *************\n", graph_name);
        printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

        int* node_feature = (int*)malloc(ND_FEATURE * num_of_nodes * sizeof(int));
        int* edge_list = (int*)malloc(2 * num_of_edges * sizeof(int));
        int* edge_attr = (int*)malloc(EDGE_ATTR * num_of_edges * sizeof(int));
        int graph_attr[2];
        graph_attr[0] = num_of_nodes;
        graph_attr[1] = num_of_edges;

        fetch_one_graph(graph_name, node_feature, edge_list, edge_attr, num_of_nodes, num_of_edges);
        
        DGN_compute_one_graph(g, node_feature, edge_list, edge_attr, graph_attr);
        all_results[g - 1] = final;
        free(node_feature);
        free(edge_list);
        free(edge_attr);
    }

    for(int g = 1; g <= 4113; g++) {
        fprintf(c_output, "g%d: %.8f\n", g, all_results[g-1]);
    }
    fclose(c_output);

    
    
    return 0;
}
