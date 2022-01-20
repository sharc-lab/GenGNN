#include "dcl.h"
#include "host.h"

aligned_vector<WT_TYPE> embedding_FC_weight_in(EMB_DIM * ND_FEATURE);
aligned_vector<WT_TYPE> embedding_FC_bias_in(EMB_DIM);
aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_weight_in(4 * 100 * 200);
aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_bias_in(4 * 100);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_weight_in(50 * 100);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_0_bias_in(50);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_weight_in(25 * 50);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_1_bias_in(25);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_weight_in(1 * 25);
aligned_vector<WT_TYPE> MLP_layer_FC_layers_2_bias_in(1);

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_DGN_compute_one_graph;
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
            OCL_CHECK(err, krnl_DGN_compute_one_graph = cl::Kernel(program, "DGN_compute_one_graph", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }


    printf("\n******* This is the HLS for DGN model *******\n");

    load_weights();
    printf("\n******* Weights loading done *******\n");

    cl::Buffer embedding_FC_weight_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        EMB_DIM * ND_FEATURE * sizeof(WT_TYPE),
        embedding_FC_weight_in.data(),
        &err);
    cl::Buffer embedding_FC_bias_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        EMB_DIM * sizeof(WT_TYPE),
        embedding_FC_bias_in.data(),
        &err);
    cl::Buffer layers_posttrans_fully_connected_0_linear_weight_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * 100 * 200 * sizeof(WT_TYPE),
        layers_posttrans_fully_connected_0_linear_weight_in.data(),
        &err);
    cl::Buffer layers_posttrans_fully_connected_0_linear_bias_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * 100 * sizeof(WT_TYPE),
        layers_posttrans_fully_connected_0_linear_bias_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_0_weight_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        50 * 100 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_0_weight_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_0_bias_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        50 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_0_bias_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_1_weight_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        25 * 50 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_1_weight_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_1_bias_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        25 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_1_bias_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_2_weight_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        1 * 25 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_2_weight_in.data(),
        &err);
    cl::Buffer MLP_layer_FC_layers_2_bias_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        1 * sizeof(WT_TYPE),
        MLP_layer_FC_layers_2_bias_in.data(),
        &err);

    int idx = 6;
    krnl_DGN_compute_one_graph.setArg(idx++, embedding_FC_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, embedding_FC_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, layers_posttrans_fully_connected_0_linear_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, layers_posttrans_fully_connected_0_linear_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_0_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_0_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_1_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_1_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_2_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_2_bias_in_buf);

    FILE* c_output = fopen("HLS_output.txt", "w+");
    char graph_name[128];
    char info_file[128];
    int num_of_nodes;
    int num_of_edges;


#ifdef GRAPH_CORA
    sprintf(info_file, "../../../graphs/graph_info/cora_info.txt");
    sprintf(graph_name, "cora");
#endif
#ifdef GRAPH_CITESEER
    sprintf(info_file, "../../../graphs/graph_info/citeseer_info.txt");
    sprintf(graph_name, "citeseer");
#endif
#ifdef GRAPH_PUBMED
    sprintf(info_file, "../../../graphs/graph_info/pubmed_info.txt");
    sprintf(graph_name, "pubmed");
#endif


    FILE* f_info = fopen(info_file, "r");
    fscanf (f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
    fclose(f_info);
    

    printf("********** Computing Graph %s *************\n", graph_name);
    printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

    aligned_vector<int> node_feature(ND_FEATURE * num_of_nodes);
    aligned_vector<WT_TYPE> node_eigen(4 * num_of_nodes);
    aligned_vector<int> edge_list(2 * num_of_edges);
    aligned_vector<int> graph_attr(3);
    graph_attr[0] = num_of_nodes;
    graph_attr[1] = num_of_edges;
    graph_attr[2] = true;

    fetch_one_graph(graph_name, node_feature, node_eigen, edge_list, num_of_nodes, num_of_edges);

    aligned_vector<int> degree_table(num_of_nodes * 2);
    aligned_vector<int> neighbor_table(num_of_edges);
    prepare_graph(num_of_nodes, num_of_edges, edge_list, degree_table, neighbor_table);

    aligned_vector<FM_TYPE> h_node_ping_dram(num_of_nodes * EMB_DIM);
    aligned_vector<FM_TYPE> h_node_pong_dram(num_of_nodes * EMB_DIM);
    aligned_vector<float> result(num_of_nodes);

    cl::Buffer node_feature_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        ND_FEATURE * num_of_nodes * sizeof(int),
        node_feature.data(),
        &err);
    cl::Buffer node_eigen_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * num_of_nodes * sizeof(WT_TYPE),
        node_eigen.data(),
        &err);
    cl::Buffer degree_table_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        num_of_nodes * 2 * sizeof(int),
        degree_table.data(),
        &err);
    cl::Buffer neighbor_table_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        num_of_edges * sizeof(int),
        neighbor_table.data(),
        &err);
    cl::Buffer graph_attr_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        3 * sizeof(int),
        graph_attr.data(),
        &err);
    cl::Buffer h_node_ping_dram_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        num_of_nodes * EMB_DIM * sizeof(FM_TYPE),
        h_node_ping_dram.data(),
        &err);
    cl::Buffer h_node_pong_dram_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        num_of_nodes * EMB_DIM * sizeof(FM_TYPE),
        h_node_pong_dram.data(),
        &err);
    cl::Buffer result_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        num_of_nodes * sizeof(float),
        result.data(),
        &err);

    krnl_DGN_compute_one_graph.setArg(0, result_buf);
    krnl_DGN_compute_one_graph.setArg(1, node_feature_buf);
    krnl_DGN_compute_one_graph.setArg(2, node_eigen_buf);
    krnl_DGN_compute_one_graph.setArg(3, degree_table_buf);
    krnl_DGN_compute_one_graph.setArg(4, neighbor_table_buf);
    krnl_DGN_compute_one_graph.setArg(5, graph_attr_buf);
    krnl_DGN_compute_one_graph.setArg(16, h_node_ping_dram_buf);
    krnl_DGN_compute_one_graph.setArg(17, h_node_pong_dram_buf);

    printf("Computing DGN ...\n");
    OCL_CHECK(err, err = q.enqueueTask(krnl_DGN_compute_one_graph));
    q.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    graph_attr[2] = false;
    cl::Buffer graph_attr2_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        3 * sizeof(int),
        graph_attr.data(),
        &err);
    krnl_DGN_compute_one_graph.setArg(5, graph_attr2_buf);
    OCL_CHECK(err, err = q.enqueueTask(krnl_DGN_compute_one_graph));
    q.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    printf("Final node predictions:\n");
    for (int nd = 0; nd < num_of_nodes; nd++) {
        printf("%.7f\n", result[nd]);
        fprintf(c_output, "%.8f\n", result[nd]);
    }
    printf("DGN computation done.\n");

    return 0;
}
