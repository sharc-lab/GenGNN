#include "dcl.h"
#include "host.h"

aligned_vector<WT_TYPE> embedding_h_atom_embedding_list_weights(9 * 119 * 100);
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

    cl::Buffer embedding_h_atom_embedding_list_weights_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        9 * 119 * 100 * sizeof(WT_TYPE),
        embedding_h_atom_embedding_list_weights.data(),
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
    krnl_DGN_compute_one_graph.setArg(idx++, embedding_h_atom_embedding_list_weights_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, layers_posttrans_fully_connected_0_linear_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, layers_posttrans_fully_connected_0_linear_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_0_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_0_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_1_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_1_bias_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_2_weight_in_buf);
    krnl_DGN_compute_one_graph.setArg(idx++, MLP_layer_FC_layers_2_bias_in_buf);

    float all_results[4113];
    FILE* c_output = fopen("HLS_output.txt", "w+");
    for(int g = 1; g <= 4113; g++ ) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;


        sprintf(info_file, "../../../graphs/graph_info/g%d_info.txt", g);
        sprintf(graph_name, "../../../graphs/graph_bin/g%d", g);


        FILE* f_info = fopen(info_file, "r");
        fscanf (f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);
        

        printf("********** Computing Graph %s *************\n", graph_name);
        printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

        aligned_vector<int> node_feature(ND_FEATURE * num_of_nodes);
        aligned_vector<WT_TYPE> node_eigen(4 * num_of_nodes);
        aligned_vector<int> edge_list(2 * num_of_edges);
        aligned_vector<int> edge_attr(EDGE_ATTR * num_of_edges);
        aligned_vector<int> graph_attr(3);
        graph_attr[0] = num_of_nodes;
        graph_attr[1] = num_of_edges;
        graph_attr[2] = g == 1;

        fetch_one_graph(g, graph_name, node_feature, node_eigen, edge_list, edge_attr, num_of_nodes, num_of_edges);

        aligned_vector<int> degree_table(num_of_nodes * 2);
        aligned_vector<int> neighbor_table(num_of_edges);
        prepare_graph(num_of_nodes, num_of_edges, edge_list, degree_table, neighbor_table);

        aligned_vector<float> result(1);

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
        cl::Buffer result_buf(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof(float),
            result.data(),
            &err);

        krnl_DGN_compute_one_graph.setArg(0, result_buf);
        krnl_DGN_compute_one_graph.setArg(1, node_feature_buf);
        krnl_DGN_compute_one_graph.setArg(2, node_eigen_buf);
        krnl_DGN_compute_one_graph.setArg(3, degree_table_buf);
        krnl_DGN_compute_one_graph.setArg(4, neighbor_table_buf);
        krnl_DGN_compute_one_graph.setArg(5, graph_attr_buf);

        printf("Computing DGN ...\n");
        OCL_CHECK(err, err = q.enqueueTask(krnl_DGN_compute_one_graph));
        q.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        printf("Final graph prediction:\n");
        printf("%.7f\n", result[0]);
        fprintf(c_output, "g%d: %.8f\n", g, result[0]);
        printf("DGN computation done.\n");
    }

    return 0;
}
