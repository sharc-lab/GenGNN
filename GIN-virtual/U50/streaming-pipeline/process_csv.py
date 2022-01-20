import csv
import glob
import os
import re

for filename in glob.glob('./opencl_summary_degree_*.csv'):
    with open(os.path.join(os.getcwd(), filename), 'r') as csvfile:
        #print(filename)
        l = re.findall(r'\d+', filename)
        print(l)

        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'GIN_compute_one_graph' in row['Profile Summary']:
                #print(row)
                for ele in row:
                    if isinstance(ele, str):
                        continue
                    time_trace = row[ele]
                    #print(time_trace)

                    total_g = time_trace[0]
                    min_time = time_trace[2]
                    ave_time = time_trace[3]
                    max_time = time_trace[4]

                    print(total_g, min_time, ave_time, max_time)