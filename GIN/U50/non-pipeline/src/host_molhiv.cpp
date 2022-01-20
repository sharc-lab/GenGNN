
#include "dcl.hpp"
#include "xcl2.hpp"

float gnn_node_mlp_1_weights[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
float gnn_node_mlp_1_bias[LAYER_NUM][MLP_1_OUT];
float gnn_node_mlp_2_weights[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
float gnn_node_mlp_2_bias[LAYER_NUM][MLP_2_OUT];
float gnn_node_embedding_table[ND_FEATURE_TOTAL][EMB_DIM];
float gnn_edge_embedding_table[EG_FEATURE_TOTAL][EMB_DIM];
float graph_pred_linear_weight[NUM_TASK][MLP_2_OUT];
float graph_pred_linear_bias[NUM_TASK];
float eps[LAYER_NUM];

// WT_TYPE gnn_node_mlp_1_weights_fixed[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
// WT_TYPE gnn_node_mlp_1_bias_fixed[LAYER_NUM][MLP_1_OUT];
// WT_TYPE gnn_node_mlp_2_weights_fixed[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
// WT_TYPE gnn_node_mlp_2_bias_fixed[LAYER_NUM][MLP_2_OUT];
// WT_TYPE gnn_node_embedding_table_fixed[ND_FEATURE_TOTAL][EMB_DIM];
// WT_TYPE gnn_edge_embedding_table_fixed[EG_FEATURE_TOTAL][EMB_DIM];
// WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK][MLP_2_OUT];
// WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK];
// WT_TYPE eps_fixed[LAYER_NUM];

std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_1_weights_fixed(LAYER_NUM * MLP_1_OUT * MLP_1_IN);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_1_bias_fixed(LAYER_NUM * MLP_1_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_2_weights_fixed(LAYER_NUM * MLP_2_OUT * MLP_2_IN);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_mlp_2_bias_fixed(LAYER_NUM * MLP_2_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_node_embedding_table_fixed(ND_FEATURE_TOTAL * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gnn_edge_embedding_table_fixed(EG_FEATURE_TOTAL * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed(NUM_TASK * MLP_2_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed(NUM_TASK);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> eps_fixed(LAYER_NUM);

void GIN_compute_one_graph();
void load_weights();
void fetch_one_graph(char* graph_name, std::vector<int, aligned_allocator<int>>* node_feature, std::vector<int, aligned_allocator<int>>* edge_list, std::vector<int, aligned_allocator<int>>* edge_attr, int num_of_nodes, int num_of_edges);


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_GIN_compute_one_graph;
    cl::CommandQueue q;
    
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_GIN_compute_one_graph = cl::Kernel(program, "GIN_compute_one_graph", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }


    printf("\n******* This is the HLS for GIN model *******\n");

    load_weights();
    printf("\n******* Weights loading done *******\n");

    cl::Buffer gnn_node_mlp_1_weights_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_1_OUT * MLP_1_IN * sizeof(WT_TYPE),
                                                gnn_node_mlp_1_weights_fixed.data(),
                                                &err);

    cl::Buffer gnn_node_mlp_1_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_1_OUT * sizeof(WT_TYPE),
                                                gnn_node_mlp_1_bias_fixed.data(),
                                                &err);

    cl::Buffer gnn_node_mlp_2_weights_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_2_OUT * MLP_2_IN * sizeof(WT_TYPE),
                                                gnn_node_mlp_2_weights_fixed.data(),
                                                &err);

    cl::Buffer gnn_node_mlp_2_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_2_OUT * sizeof(WT_TYPE),
                                                gnn_node_mlp_2_bias_fixed.data(),
                                                &err);

    cl::Buffer gnn_node_embedding_table_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                ND_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
                                                gnn_node_embedding_table_fixed.data(),
                                                &err);

    cl::Buffer gnn_edge_embedding_table_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                EG_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
                                                gnn_edge_embedding_table_fixed.data(),
                                                &err);

    cl::Buffer graph_pred_linear_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * MLP_2_OUT * sizeof(WT_TYPE),
                                                graph_pred_linear_weight_fixed.data(),
                                                &err);

    cl::Buffer graph_pred_linear_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * sizeof(WT_TYPE),
                                                graph_pred_linear_bias_fixed.data(),
                                                &err);

    cl::Buffer eps_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * sizeof(WT_TYPE),
                                                eps_fixed.data(),
                                                &err);


    int idx = 5;
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_node_mlp_1_weights_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_node_mlp_1_bias_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_node_mlp_2_weights_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_node_mlp_2_bias_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_node_embedding_table_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, gnn_edge_embedding_table_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, graph_pred_linear_weight_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, graph_pred_linear_bias_fixed_in);
    krnl_GIN_compute_one_graph.setArg(idx++, eps_fixed_in);
    

    float all_results[4113];
    FILE* c_output = fopen("HLS_output.txt", "w+");
    int is_first = 1;
    for(int g = 1; g <= 1000; g++ ) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

	//sprintf(info_file, "../../../../graph_info/g%d_info.txt", g);
	//sprintf(graph_name, "../../../../graph_bin/g%d", g);
	//
	//sprintf(info_file, "gtest_info.txt");
	//sprintf(graph_name, "gtest");
	//
	sprintf(info_file, "/nethome/chao33/test_graphs/random_graphs/g_rand_%d_info.txt", g);
	sprintf(graph_name, "/nethome/chao33/test_graphs/random_graphs/g_rand_%d", g);

        FILE* f_info = fopen(info_file, "r");
        fscanf (f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
	fclose(f_info);

printf("********** Computing Graph %s *************\n", graph_name);
printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

        std::vector<int, aligned_allocator<int>> node_feature(ND_FEATURE * MAX_NODE);
        std::vector<int, aligned_allocator<int>> edge_list(2 * MAX_EDGE);
        std::vector<int, aligned_allocator<int>> edge_attr(EDGE_ATTR * MAX_EDGE);
        std::vector<int, aligned_allocator<int>> graph_attr(3);
        std::vector<FM_TYPE, aligned_allocator<FM_TYPE>> task_out(NUM_TASK);

        graph_attr[0] = num_of_nodes;
        graph_attr[1] = num_of_edges;
	graph_attr[2] = is_first;

        fetch_one_graph(graph_name, &node_feature, &edge_list, &edge_attr, num_of_nodes, num_of_edges);


        cl::Buffer node_feature_in( context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            ND_FEATURE * MAX_NODE * sizeof(int),
                                            node_feature.data(),
                                            &err);

        cl::Buffer edge_list_in( context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            2 * MAX_EDGE * sizeof(int),
                                            edge_list.data(),
                                            &err);

        cl::Buffer edge_attr_in( context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            EDGE_ATTR * MAX_EDGE * sizeof(int),
                                            edge_attr.data(),
                                            &err);

        cl::Buffer graph_attr_in( context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            3 * sizeof(int),
                                            graph_attr.data(),
                                            &err);

        cl::Buffer task_result( context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            NUM_TASK * sizeof(FM_TYPE),
                                            task_out.data(),
                                            &err);

        krnl_GIN_compute_one_graph.setArg(0, node_feature_in);
        krnl_GIN_compute_one_graph.setArg(1, edge_list_in);
        krnl_GIN_compute_one_graph.setArg(2, edge_attr_in);
        krnl_GIN_compute_one_graph.setArg(3, graph_attr_in);
        krnl_GIN_compute_one_graph.setArg(4, task_result);

        OCL_CHECK(err, err = q.enqueueTask(krnl_GIN_compute_one_graph));
        q.enqueueMigrateMemObjects({task_result}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        
	printf("Final graph prediction:\n");
        for(int tsk = 0; tsk < NUM_TASK; tsk++) {
            printf("%.7f\n", task_out[tsk].to_float());
        }
        printf("GIN computation done.\n");

	is_first = 0;
    }

    
    return 0;
}
