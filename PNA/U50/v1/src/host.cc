#include "host.h"
#include "dcl.h"

aligned_vector<WT_TYPE> node_emb_atom_embedding_list_0_weight_fixed_in(119 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_1_weight_fixed_in(4 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_2_weight_fixed_in(12 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_3_weight_fixed_in(12 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_4_weight_fixed_in(10 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_5_weight_fixed_in(6 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_6_weight_fixed_in(6 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_7_weight_fixed_in(2 * 80);
aligned_vector<WT_TYPE> node_emb_atom_embedding_list_8_weight_fixed_in(2 * 80);

aligned_vector<WT_TYPE> mlp_0_weight_fixed_in(40 * 80);
aligned_vector<WT_TYPE> mlp_0_bias_fixed_in(40);
aligned_vector<WT_TYPE> mlp_2_weight_fixed_in(20 * 40);
aligned_vector<WT_TYPE> mlp_2_bias_fixed_in(20);
aligned_vector<WT_TYPE> mlp_4_weight_fixed_in(1 * 20);
aligned_vector<WT_TYPE> mlp_4_bias_fixed_in(1);

aligned_vector<WT_TYPE> convs_ALL_post_nn_0_weight_fixed_in(4 * 80 * 960);
aligned_vector<WT_TYPE> convs_ALL_post_nn_0_bias_fixed_in(4 * 80);

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_PNA_compute_one_graph;
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
            OCL_CHECK(err, krnl_PNA_compute_one_graph = cl::Kernel(program, "PNA_compute_one_graph", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    printf("\n******* This is the HLS for PNA model *******\n");

    load_weights();

    printf("\n******* Weights loading done *******\n");

    cl::Buffer node_emb_atom_embedding_list_0_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        119 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_0_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_1_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_1_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_2_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        12 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_2_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_3_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        12 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_3_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_4_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        10 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_4_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_5_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        6 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_5_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_6_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        6 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_6_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_7_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        2 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_7_weight_fixed_in.data(),
        &err);
    cl::Buffer node_emb_atom_embedding_list_8_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        2 * 80 * sizeof(WT_TYPE),
        node_emb_atom_embedding_list_8_weight_fixed_in.data(),
        &err);

    cl::Buffer mlp_0_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        40 * 80 * sizeof(WT_TYPE),
        mlp_0_weight_fixed_in.data(),
        &err);
    cl::Buffer mlp_0_bias_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        40 * sizeof(WT_TYPE),
        mlp_0_bias_fixed_in.data(),
        &err);
    cl::Buffer mlp_2_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        20 * 40 * sizeof(WT_TYPE),
        mlp_2_weight_fixed_in.data(),
        &err);
    cl::Buffer mlp_2_bias_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        20 * sizeof(WT_TYPE),
        mlp_2_bias_fixed_in.data(),
        &err);
    cl::Buffer mlp_4_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        1 * 20 * sizeof(WT_TYPE),
        mlp_4_weight_fixed_in.data(),
        &err);
    cl::Buffer mlp_4_bias_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        1 * sizeof(WT_TYPE),
        mlp_4_bias_fixed_in.data(),
        &err);

    cl::Buffer convs_ALL_post_nn_0_weight_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * 80 * 960 * sizeof(WT_TYPE),
        convs_ALL_post_nn_0_weight_fixed_in.data(),
        &err);
    cl::Buffer convs_ALL_post_nn_0_bias_fixed_in_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        4 * 80 * sizeof(WT_TYPE),
        convs_ALL_post_nn_0_bias_fixed_in.data(),
        &err);

    int index = 4;
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_0_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_1_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_2_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_3_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_4_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_5_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_6_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_7_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, node_emb_atom_embedding_list_8_weight_fixed_in_buf);

    krnl_PNA_compute_one_graph.setArg(index++, mlp_0_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, mlp_0_bias_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, mlp_2_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, mlp_2_bias_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, mlp_4_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, mlp_4_bias_fixed_in_buf);

    krnl_PNA_compute_one_graph.setArg(index++, convs_ALL_post_nn_0_weight_fixed_in_buf);
    krnl_PNA_compute_one_graph.setArg(index++, convs_ALL_post_nn_0_bias_fixed_in_buf);

    float all_results[4113];
    FILE *c_output = fopen("HLS_output.txt", "w+");
    for (int g = 1; g <= 4113; g++) {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

        sprintf(info_file, "../../../graphs/graph_info/g%d_info.txt", g);
        sprintf(graph_name, "../../../graphs/graph_bin/g%d", g);

        FILE *f_info = fopen(info_file, "r");
        fscanf(f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);

        printf("********** Computing Graph %s *************\n", graph_name);
        printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);

        aligned_vector<int> node_feature(ND_FEATURE * num_of_nodes);
        aligned_vector<int> edge_list(2 * num_of_edges);
        aligned_vector<int> edge_attr(EDGE_ATTR * num_of_edges);
        aligned_vector<int> graph_attr(3);
        graph_attr[0] = num_of_nodes;
        graph_attr[1] = num_of_edges;
        graph_attr[2] = g == 1;

        aligned_vector<FM_TYPE> task_tb(NUM_TASK);

        fetch_one_graph(graph_name, node_feature, edge_list, edge_attr, num_of_nodes, num_of_edges);

        cl::Buffer task_tb_buf(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            NUM_TASK * sizeof(FM_TYPE),
            task_tb.data(),
            &err);
        cl::Buffer node_feature_buf(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            ND_FEATURE * num_of_nodes * sizeof(int),
            node_feature.data(),
            &err);
        cl::Buffer edge_list_buf(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            2 * num_of_edges * sizeof(int),
            edge_list.data(),
            &err);
        cl::Buffer graph_attr_buf(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            3 * sizeof(int),
            graph_attr.data(),
            &err);

        krnl_PNA_compute_one_graph.setArg(0, task_tb_buf);
        krnl_PNA_compute_one_graph.setArg(1, node_feature_buf);
        krnl_PNA_compute_one_graph.setArg(2, edge_list_buf);
        krnl_PNA_compute_one_graph.setArg(3, graph_attr_buf);

        printf("Computing PNA ...\n");
        OCL_CHECK(err, err = q.enqueueTask(krnl_PNA_compute_one_graph));
        q.enqueueMigrateMemObjects({task_tb_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

        printf("Final graph prediction:\n");
        printf("%.7f\n", task_tb[0].to_float());
        fprintf(c_output, "g%d: %.8f\n", g, task_tb[0].to_float());
        printf("PNA computation done.\n");
    }

    return 0;
}