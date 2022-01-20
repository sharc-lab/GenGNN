#include <stdlib.h>
#include <stdio.h>
#include "dcl.h"
#include "host.h"

void load_weights()
{
    printf("Loading weights for DGN ...\n");

    FILE* f;
    f = fopen("dgn_ep1_noBN_dim100.weights.all.bin", "rb");
    fseek(f, 0*sizeof(float), SEEK_SET);	fseek(f, 0*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_0_weight_float = new float[11900];
    fread(embedding_h_atom_embedding_list_0_weight_float, sizeof(float), 11900, f);
    for (int i = 0; i < 11900; i++) embedding_h_atom_embedding_list_weights[0 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_0_weight_float[i]);
    delete embedding_h_atom_embedding_list_0_weight_float;

    fseek(f, 11900*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_1_weight_float = new float[400];
    fread(embedding_h_atom_embedding_list_1_weight_float, sizeof(float), 400, f);
    for (int i = 0; i < 400; i++) embedding_h_atom_embedding_list_weights[1 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_1_weight_float[i]);
    delete embedding_h_atom_embedding_list_1_weight_float;

    fseek(f, 12300*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_2_weight_float = new float[1200];
    fread(embedding_h_atom_embedding_list_2_weight_float, sizeof(float), 1200, f);
    for (int i = 0; i < 1200; i++) embedding_h_atom_embedding_list_weights[2 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_2_weight_float[i]);
    delete embedding_h_atom_embedding_list_2_weight_float;

    fseek(f, 13500*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_3_weight_float = new float[1200];
    fread(embedding_h_atom_embedding_list_3_weight_float, sizeof(float), 1200, f);
    for (int i = 0; i < 1200; i++) embedding_h_atom_embedding_list_weights[3 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_3_weight_float[i]);
    delete embedding_h_atom_embedding_list_3_weight_float;

    fseek(f, 14700*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_4_weight_float = new float[1000];
    fread(embedding_h_atom_embedding_list_4_weight_float, sizeof(float), 1000, f);
    for (int i = 0; i < 1000; i++) embedding_h_atom_embedding_list_weights[4 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_4_weight_float[i]);
    delete embedding_h_atom_embedding_list_4_weight_float;

    fseek(f, 15700*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_5_weight_float = new float[600];
    fread(embedding_h_atom_embedding_list_5_weight_float, sizeof(float), 600, f);
    for (int i = 0; i < 600; i++) embedding_h_atom_embedding_list_weights[5 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_5_weight_float[i]);
    delete embedding_h_atom_embedding_list_5_weight_float;

    fseek(f, 16300*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_6_weight_float = new float[600];
    fread(embedding_h_atom_embedding_list_6_weight_float, sizeof(float), 600, f);
    for (int i = 0; i < 600; i++) embedding_h_atom_embedding_list_weights[6 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_6_weight_float[i]);
    delete embedding_h_atom_embedding_list_6_weight_float;

    fseek(f, 16900*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_7_weight_float = new float[200];
    fread(embedding_h_atom_embedding_list_7_weight_float, sizeof(float), 200, f);
    for (int i = 0; i < 200; i++) embedding_h_atom_embedding_list_weights[7 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_7_weight_float[i]);
    delete embedding_h_atom_embedding_list_7_weight_float;

    fseek(f, 17100*sizeof(float), SEEK_SET);
    float *embedding_h_atom_embedding_list_8_weight_float = new float[200];
    fread(embedding_h_atom_embedding_list_8_weight_float, sizeof(float), 200, f);
    for (int i = 0; i < 200; i++) embedding_h_atom_embedding_list_weights[8 * 119 * 100 + i] = WT_TYPE(embedding_h_atom_embedding_list_8_weight_float[i]);
    delete embedding_h_atom_embedding_list_8_weight_float;

    fseek(f, 17300*sizeof(float), SEEK_SET);
    float *layers_0_posttrans_fully_connected_0_linear_weight_float = new float[20000];
    fread(layers_0_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);
    for (int i = 0; i < 20000; i++) layers_posttrans_fully_connected_0_linear_weight_in[0 * 100 * 200 + i] = WT_TYPE(layers_0_posttrans_fully_connected_0_linear_weight_float[i]);
    delete layers_0_posttrans_fully_connected_0_linear_weight_float;

    fseek(f, 37300*sizeof(float), SEEK_SET);
    float *layers_0_posttrans_fully_connected_0_linear_bias_float = new float[100];
    fread(layers_0_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
    for (int i = 0; i < 100; i++) layers_posttrans_fully_connected_0_linear_bias_in[0 * 100 + i] = WT_TYPE(layers_0_posttrans_fully_connected_0_linear_bias_float[i]);
    delete layers_0_posttrans_fully_connected_0_linear_bias_float;

    
    fseek(f, 37400*sizeof(float), SEEK_SET);
    float *layers_1_posttrans_fully_connected_0_linear_weight_float = new float[20000];
    fread(layers_1_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);
    for (int i = 0; i < 20000; i++) layers_posttrans_fully_connected_0_linear_weight_in[1 * 100 * 200 + i] = WT_TYPE(layers_1_posttrans_fully_connected_0_linear_weight_float[i]);
    delete layers_1_posttrans_fully_connected_0_linear_weight_float;

    fseek(f, 57400*sizeof(float), SEEK_SET);
    float *layers_1_posttrans_fully_connected_0_linear_bias_float = new float[100];
    fread(layers_1_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
    for (int i = 0; i < 100; i++) layers_posttrans_fully_connected_0_linear_bias_in[1 * 100 + i] = WT_TYPE(layers_1_posttrans_fully_connected_0_linear_bias_float[i]);
    delete layers_1_posttrans_fully_connected_0_linear_bias_float;
    
    fseek(f, 57500*sizeof(float), SEEK_SET);
    float *layers_2_posttrans_fully_connected_0_linear_weight_float = new float[20000];
    fread(layers_2_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);
    for (int i = 0; i < 20000; i++) layers_posttrans_fully_connected_0_linear_weight_in[2 * 100 * 200 + i] = WT_TYPE(layers_2_posttrans_fully_connected_0_linear_weight_float[i]);
    delete layers_2_posttrans_fully_connected_0_linear_weight_float;

    fseek(f, 77500*sizeof(float), SEEK_SET);
    float *layers_2_posttrans_fully_connected_0_linear_bias_float = new float[100];
    fread(layers_2_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
    for (int i = 0; i < 100; i++) layers_posttrans_fully_connected_0_linear_bias_in[2 * 100 + i] = WT_TYPE(layers_2_posttrans_fully_connected_0_linear_bias_float[i]);
    delete layers_2_posttrans_fully_connected_0_linear_bias_float;
    
    fseek(f, 77600*sizeof(float), SEEK_SET);
    float *layers_3_posttrans_fully_connected_0_linear_weight_float = new float[20000];
    fread(layers_3_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);
    for (int i = 0; i < 20000; i++) layers_posttrans_fully_connected_0_linear_weight_in[3 * 100 * 200 + i] = WT_TYPE(layers_3_posttrans_fully_connected_0_linear_weight_float[i]);
    delete layers_3_posttrans_fully_connected_0_linear_weight_float;

    fseek(f, 97600*sizeof(float), SEEK_SET);
    float *layers_3_posttrans_fully_connected_0_linear_bias_float = new float[100];
    fread(layers_3_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
    for (int i = 0; i < 100; i++) layers_posttrans_fully_connected_0_linear_bias_in[3 * 100 + i] = WT_TYPE(layers_3_posttrans_fully_connected_0_linear_bias_float[i]);
    delete layers_3_posttrans_fully_connected_0_linear_bias_float;

    fseek(f, 97700*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_0_weight_float = new float[5000];
    fread(MLP_layer_FC_layers_0_weight_float, sizeof(float), 5000, f);
    for (int i = 0; i < 5000; i++) MLP_layer_FC_layers_0_weight_in[i] = WT_TYPE(MLP_layer_FC_layers_0_weight_float[i]);
    delete MLP_layer_FC_layers_0_weight_float;

    fseek(f, 102700*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_0_bias_float = new float[50];
    fread(MLP_layer_FC_layers_0_bias_float, sizeof(float), 50, f);
    for (int i = 0; i < 50; i++) MLP_layer_FC_layers_0_bias_in[i] = WT_TYPE(MLP_layer_FC_layers_0_bias_float[i]);
    delete MLP_layer_FC_layers_0_bias_float;
    
    fseek(f, 102750*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_1_weight_float = new float[1250];
    fread(MLP_layer_FC_layers_1_weight_float, sizeof(float), 1250, f);
    for (int i = 0; i < 1250; i++) MLP_layer_FC_layers_1_weight_in[i] = WT_TYPE(MLP_layer_FC_layers_1_weight_float[i]);
    delete MLP_layer_FC_layers_1_weight_float;

    fseek(f, 104000*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_1_bias_float = new float[25];
    fread(MLP_layer_FC_layers_1_bias_float, sizeof(float), 25, f);
    for (int i = 0; i < 25; i++) MLP_layer_FC_layers_1_bias_in[i] = WT_TYPE(MLP_layer_FC_layers_1_bias_float[i]);
    delete MLP_layer_FC_layers_1_bias_float;

    fseek(f, 104025*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_2_weight_float = new float[25];
    fread(MLP_layer_FC_layers_2_weight_float, sizeof(float), 25, f);
    for (int i = 0; i < 25; i++) MLP_layer_FC_layers_2_weight_in[i] = WT_TYPE(MLP_layer_FC_layers_2_weight_float[i]);
    delete MLP_layer_FC_layers_2_weight_float;

    fseek(f, 104050*sizeof(float), SEEK_SET);
    float *MLP_layer_FC_layers_2_bias_float = new float[1];
    fread(MLP_layer_FC_layers_2_bias_float, sizeof(float), 1, f);
    for (int i = 0; i < 1; i++) MLP_layer_FC_layers_2_bias_in[i] = WT_TYPE(MLP_layer_FC_layers_2_bias_float[i]);
    delete MLP_layer_FC_layers_2_bias_float;

    fclose(f);

}
void fetch_one_graph(int g, char* graph_name, aligned_vector<int>& node_feature, aligned_vector<WT_TYPE>& node_eigen, aligned_vector<int>& edge_list, aligned_vector<int>& edge_attr, int num_of_nodes, int num_of_edges)
{
    printf("Loading graph ...\n");
        
    FILE* f;

    char f_node_feature[128];
    char f_edge_list[128];
    char f_edge_attr[128];
    char f_node_eigen[128];

    sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
    sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
    sprintf(f_edge_attr, "%s_edge_attr.bin", graph_name);
    sprintf(f_node_eigen, "eig/g%d.txt", g);
    
    
    f = fopen(f_node_feature, "rb");
    fread(node_feature.data(), sizeof(int), num_of_nodes * ND_FEATURE, f);
    fclose(f);


    f = fopen(f_edge_list, "rb");
    fread(edge_list.data(), sizeof(int), 2 * num_of_edges, f);
    fclose(f);


    f = fopen(f_edge_attr, "rb");
    fread(edge_attr.data(), sizeof(int), EDGE_ATTR * num_of_edges, f);
    fclose(f);

    f = fopen(f_node_eigen, "r");
    float node_eigen_float[4];
    fscanf(f, "tensor([[%e, %e,%e,%e],\n", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
    for (int i = 0; i < 4; i++) node_eigen[i] = WT_TYPE(node_eigen_float[i]);
    for (int nd = 1; nd < num_of_nodes - 1; nd++)
    {
        fscanf(f, "[%e, %e,%e,%e],\n", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
        for (int i = 0; i < 4; i++) node_eigen[(nd * 4) + i] = WT_TYPE(node_eigen_float[i]);
    }
    fscanf(f, "[%e, %e,%e,%e]])", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
    for (int i = 0; i < 4; i++) node_eigen[((num_of_nodes-1) * 4) + i] = WT_TYPE(node_eigen_float[i]);
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
#endif
}

void prepare_graph(
    int num_of_nodes,
    int num_of_edges,
    aligned_vector<int>& edge_list,
    aligned_vector<int>& degree_table,
    aligned_vector<int>& neighbor_table
)
{
    printf("Preparing graph ...\n");

    int neighbor_table_idxs[num_of_nodes];
    int edge_list_len = num_of_edges * 2;

    for (int i = 0; i < num_of_nodes; i++)
    {
        degree_table[i * 2] = 0;
        neighbor_table_idxs[i] = 0;
    }

    for (int i = 1; i < edge_list_len; i += 2)
    {
        int v = edge_list[i];
        degree_table[v * 2]++;
    }

    int acc = 0;
    for (int i = 0; i < num_of_nodes; i++)
    {
        int degree = degree_table[i * 2];
        degree_table[i * 2 + 1] = acc;
        acc += degree;
    }

    for (int i = 0; i < edge_list_len; i += 2)
    {
        int u = edge_list[i];
        int v = edge_list[i + 1];
        int row_idx = degree_table[v * 2 + 1];
        int col_idx = neighbor_table_idxs[v];
        int e = row_idx + col_idx;
        neighbor_table[e] = u;
        neighbor_table_idxs[v] = col_idx + 1;
    }
}
