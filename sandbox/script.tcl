open_project project_1
set_top top

add_files test.cpp

open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}

csynth_design
#export_design -evaluate verilog -format ip_catalog
export_design -format ip_catalog

exit
