#include <stdio.h>
#include <stdlib.h>
#include "dcl.hpp"
#include "xcl2.hpp"

// global weights
float graph_pred_linear_weight[NUM_TASK][FEATURE_OUT];
float graph_pred_linear_bias[NUM_TASK];
float gat_net_scoring_fn_target[LAYER_NUM][HEAD_NUM][FEATURE_OUT];
float gat_net_scoring_fn_source[LAYER_NUM][HEAD_NUM][FEATURE_OUT];
float gat_net_0_linear_proj_weight[HEAD_NUM * FEATURE_OUT][FEATURE_IN];
float gat_net_1_linear_proj_weight[LAYER_NUM - 1][HEAD_NUM * FEATURE_OUT][HEAD_NUM * FEATURE_OUT];
float gat_net_0_skip_proj_weight[HEAD_NUM * FEATURE_OUT][FEATURE_IN];
float gat_net_1_skip_proj_weight[LAYER_NUM - 1][HEAD_NUM * FEATURE_OUT][HEAD_NUM * FEATURE_OUT];

// WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * FEATURE_OUT];
// WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK];
// WT_TYPE gat_net_scoring_fn_target_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT];
// WT_TYPE gat_net_scoring_fn_source_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT];
// WT_TYPE gat_net_linear_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT];
// WT_TYPE gat_net_skip_proj_weight_fixed[LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT];

std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_weight_fixed(NUM_TASK * FEATURE_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> graph_pred_linear_bias_fixed(NUM_TASK);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_scoring_fn_target_fixed(LAYER_NUM * HEAD_NUM * FEATURE_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_scoring_fn_source_fixed(LAYER_NUM * HEAD_NUM * FEATURE_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_linear_proj_weight_fixed(LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT);
std::vector<WT_TYPE, aligned_allocator<WT_TYPE>> gat_net_skip_proj_weight_fixed(LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT);

void GAT_compute_one_graph();
void load_weights();
void fetch_one_graph(char* graph_name, std::vector<int, aligned_allocator<int>>* node_feature, std::vector<int, aligned_allocator<int>>* edge_list, int num_of_nodes, int num_of_edges);

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_GAT_compute_one_graph;
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
            OCL_CHECK(err, krnl_GAT_compute_one_graph = cl::Kernel(program, "GAT_compute_one_graph", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }


    printf("\n******* This is the HLS file for GAT model *******\n");

    load_weights();

    cl::Buffer graph_pred_linear_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * FEATURE_OUT * sizeof(WT_TYPE),
                                                graph_pred_linear_weight_fixed.data(),
                                                &err);
    
    cl::Buffer graph_pred_linear_bias_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                NUM_TASK * sizeof(WT_TYPE),
                                                graph_pred_linear_bias_fixed.data(),
                                                &err);

    cl::Buffer gat_net_scoring_fn_target_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * HEAD_NUM * FEATURE_OUT * sizeof(WT_TYPE),
                                                gat_net_scoring_fn_target_fixed.data(),
                                                &err);

    cl::Buffer gat_net_scoring_fn_source_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * HEAD_NUM * FEATURE_OUT * sizeof(WT_TYPE),
                                                gat_net_scoring_fn_source_fixed.data(),
                                                &err);

    cl::Buffer gat_net_linear_proj_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT * sizeof(WT_TYPE),
                                                gat_net_linear_proj_weight_fixed.data(),
                                                &err);

    cl::Buffer gat_net_skip_proj_weight_fixed_in( context,
                                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                LAYER_NUM * HEAD_NUM * FEATURE_OUT * HEAD_NUM * FEATURE_OUT * sizeof(WT_TYPE),
                                                gat_net_skip_proj_weight_fixed.data(),
                                                &err);

    int idx = 4;
    krnl_GAT_compute_one_graph.setArg(idx++, graph_pred_linear_weight_fixed_in);
    krnl_GAT_compute_one_graph.setArg(idx++, graph_pred_linear_bias_fixed_in);
    krnl_GAT_compute_one_graph.setArg(idx++, gat_net_scoring_fn_target_fixed_in);
    krnl_GAT_compute_one_graph.setArg(idx++, gat_net_scoring_fn_source_fixed_in);
    krnl_GAT_compute_one_graph.setArg(idx++, gat_net_linear_proj_weight_fixed_in);
    krnl_GAT_compute_one_graph.setArg(idx++, gat_net_skip_proj_weight_fixed_in);

    float all_results[4113];
    int is_first = 1;
    FILE* c_output = fopen("HLS_output.txt", "w+");
    for(int g = 1; g <= 4113; g++ ) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

        sprintf(info_file, "../../../graphs/graph_info/g%d_info.txt", g);
        sprintf(graph_name, "../../../graphs/graph_bin/g%d", g);
        
        FILE* f_info = fopen(info_file, "r");
        fscanf(f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);
        
        printf("********** Computing Graph %s *************\n", graph_name);
        printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

        // int node_feature[100000];
        // int edge_list[100000];
        // int graph_attr[3];
        std::vector<int, aligned_allocator<int>> node_feature(ND_FEATURE * MAX_NODE);
        std::vector<int, aligned_allocator<int>> edge_list(2 * MAX_EDGE);
        std::vector<int, aligned_allocator<int>> graph_attr(3);
        std::vector<FM_TYPE, aligned_allocator<FM_TYPE>> task_out(NUM_TASK);

        graph_attr[0] = num_of_nodes;
        graph_attr[1] = num_of_edges;
        graph_attr[2] = is_first;

        // FM_TYPE task_tb[NUM_TASK];

        fetch_one_graph(graph_name, &node_feature, &edge_list, num_of_nodes, num_of_edges);
        
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

        krnl_GAT_compute_one_graph.setArg(0, node_feature_in);
        krnl_GAT_compute_one_graph.setArg(1, edge_list_in);
        krnl_GAT_compute_one_graph.setArg(2, graph_attr_in);
        krnl_GAT_compute_one_graph.setArg(3, task_result);

        OCL_CHECK(err, err = q.enqueueTask(krnl_GAT_compute_one_graph));
        q.enqueueMigrateMemObjects({task_result}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();
        
        //all_results[g-1] = task_tb[0].to_float();

        //free(node_feature);
        //free(edge_list);

        is_first = 0;
    }
    /*
    for(int g = 1; g <= 2; g++) {
        fprintf(c_output, "g%d: %.8f\n", g, all_results[g-1]);
    }
    */
    fclose(c_output);
    
    return 0;
}
