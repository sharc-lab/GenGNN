
#include "dcl.hpp"
#include "xcl2.hpp"


std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> node_embedding_weight_fixed(ND_FEATURE_TOTAL * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> edge_embedding_weight_fixed(EG_FEATURE_TOTAL * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_weight_fixed(LAYER_NUM * MLP_0_OUT * MLP_0_IN);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_bias_fixed(LAYER_NUM * MLP_0_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> convs_root_emb_weight_fixed(LAYER_NUM * MLP_0_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_weight_fixed(LAYER_NUM * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_bias_fixed(LAYER_NUM * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_mean_fixed(LAYER_NUM * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> bn_var_fixed(LAYER_NUM * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed(NUM_TASK * EMB_DIM);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed(NUM_TASK);

void GCN_compute_one_graph();
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
    cl::Kernel krnl_GCN_compute_one_graph;
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
            OCL_CHECK(err, krnl_GCN_compute_one_graph = cl::Kernel(program, "GCN_compute_one_graph", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }


    printf("\n******* This is the HLS for GCN model *******\n");

    load_weights();
    printf("\n******* Weights loading done *******\n");

    cl::Buffer node_embedding_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                ND_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
                                                node_embedding_weight_fixed.data(),
                                                &err);

    cl::Buffer edge_embedding_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                EG_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
                                                edge_embedding_weight_fixed.data(),
                                                &err);

    cl::Buffer convs_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_0_OUT * MLP_0_IN * sizeof(WT_TYPE),
                                                convs_weight_fixed.data(),
                                                &err);

    cl::Buffer convs_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_0_OUT * sizeof(WT_TYPE),
                                                convs_bias_fixed.data(),
                                                &err);

    cl::Buffer convs_root_emb_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * MLP_0_OUT * sizeof(WT_TYPE),
                                                convs_root_emb_weight_fixed.data(),
                                                &err);

    cl::Buffer bn_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * EMB_DIM * sizeof(WT_TYPE),
                                                bn_weight_fixed.data(),
                                                &err);

    cl::Buffer bn_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * EMB_DIM * sizeof(WT_TYPE),
                                                bn_bias_fixed.data(),
                                                &err);

    cl::Buffer bn_mean_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * EMB_DIM * sizeof(WT_TYPE),
                                                bn_mean_fixed.data(),
                                                &err);

    cl::Buffer bn_var_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * EMB_DIM * sizeof(WT_TYPE),
                                                bn_var_fixed.data(),
                                                &err);

    cl::Buffer graph_pred_linear_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * EMB_DIM * sizeof(WT_TYPE),
                                                graph_pred_linear_weight_fixed.data(),
                                                &err);

    cl::Buffer graph_pred_linear_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * sizeof(WT_TYPE),
                                                graph_pred_linear_bias_fixed.data(),
                                                &err);


    int idx = 5;
    krnl_GCN_compute_one_graph.setArg(idx++, convs_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, convs_bias_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, convs_root_emb_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, bn_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, bn_bias_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, bn_mean_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, bn_var_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, node_embedding_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, edge_embedding_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, graph_pred_linear_weight_fixed_in);
    krnl_GCN_compute_one_graph.setArg(idx++, graph_pred_linear_bias_fixed_in);
    

    float all_results[4113];
    FILE* c_output = fopen("HLS_output.txt", "w+");
    int is_first = 1;
    for(int g = 1; g <= 4112; g++ ) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

	sprintf(info_file, "../../../../graph_info/g%d_info.txt", g);
	sprintf(graph_name, "../../../../graph_bin/g%d", g);
	
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
                                            NUM_TASK * sizeof(WT_TYPE),
                                            task_out.data(),
                                            &err);

        krnl_GCN_compute_one_graph.setArg(0, node_feature_in);
        krnl_GCN_compute_one_graph.setArg(1, edge_list_in);
        krnl_GCN_compute_one_graph.setArg(2, edge_attr_in);
        krnl_GCN_compute_one_graph.setArg(3, graph_attr_in);
        krnl_GCN_compute_one_graph.setArg(4, task_result);

        OCL_CHECK(err, err = q.enqueueTask(krnl_GCN_compute_one_graph));
        q.enqueueMigrateMemObjects({task_result}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        
	printf("Final graph prediction:\n");
        for(int tsk = 0; tsk < NUM_TASK; tsk++) {
            printf("%.7f\n", task_out[tsk].to_float());
        }
        printf("GCN computation done.\n");

	is_first = 0;
    }

    
    return 0;
}
