open_project -reset PNA_HLS_proj

add_files main.cc
add_files dcl.h
add_files load_weights_graph.cc
add_files PNA_compute.cc
add_files pna_conv_bias_dim80.bin
add_files pna_conv_weights_dim80.bin
add_files pna_ep1_nd_embed_dim80.bin
add_files pna_ep1_noBN_dim80.weights.all.bin

set_top PNA_compute_one_graph

open_solution "solution1" -flow_target vivado
set_part {xcvu11p-flga2577-1-e}
create_clock -period 4 -name default

# csim_design
csynth_design
# cosim_design