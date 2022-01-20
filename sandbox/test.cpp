

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <hls_stream.h>


void PE(hls::stream<int> &weights_in, hls::stream<int> &weights_out, int data_in[10], int data_out_local[20], int d_out, int final)
{
	data_out_local[d_out] = 0;
    for(int i = 0; i < 10; i++) {
#pragma HLS pipeline
    	int w = weights_in.read();
    	data_out_local[d_out] += w * data_in[i];
        if( final != 1 ) weights_out.write(w);
    }
}


void load_weight(int weights[20][10], hls::stream<int> &weight_buf, int d_out)
{
    for(int i = 0; i < 10; i++)
    {
#pragma HLS pipeline
    	weight_buf.write(weights[d_out][i]);
    }
}



void PE_dataflow(int weights[20][10], int data_in_local[4][10], int data_out_local[4][20], int d_out)
{
#pragma HLS dataflow

#pragma HLS array_partition variable=data_in_local dim=1 complete
#pragma HLS array_partition variable=data_out_local dim=1 complete

		hls::stream<int> weight_buf;
		hls::stream<int> weights_stream[4];

		load_weight(weights, weight_buf, d_out);
		PE(weight_buf, weights_stream[0], data_in_local[0], data_out_local[0], d_out, 0);
		for(int n = 1; n <= 2; n++) {
#pragma HLS unroll
			PE(weights_stream[n-1], weights_stream[n], data_in_local[n], data_out_local[n], d_out, 0);
		}
		PE(weights_stream[2], weights_stream[3], data_in_local[3], data_out_local[3], d_out, 1);
}




// 10 in_dim -> 20 out_dim
void top(int weights[20][10], int data_in[4][10], int data_out[4][20])
{
#pragma HLS INTERFACE m_axi depth=200 port=weights  offset=slave 
#pragma HLS INTERFACE m_axi depth=40  port=data_in  offset=slave 
#pragma HLS INTERFACE m_axi depth=80  port=data_out offset=slave 

	int data_in_local[4][10];
	int data_out_local[4][20];

//#pragma HLS bind_storage variable=data_in_local type=RAM_2P impl=bram
//#pragma HLS bind_storage variable=data_out_local type=RAM_2P impl=bram


	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 10; j++) {
			data_in_local[i][j] = data_in[i][j];
		}
	}

	loop_d_out: for(int d_out = 0; d_out < 20; d_out++) {
		PE_dataflow( weights, data_in_local, data_out_local, d_out);
	}


	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 20; j++) {
			data_out[i][j] = data_out_local[i][j];
		}
	}
}


void test();


int main()
{
    int weights[20][10];
    int data_in[4][10];
    int data_out[4][20];
    int data_out_ref[4][20];

    memset(data_out, 0, 4 * 20 * sizeof(int));
    memset(data_out_ref, 0, 4 * 20 * sizeof(int));

    for(int i = 0; i < 20; i++) {
        for(int j = 0; j < 10; j++) {
            weights[i][j] = i + j;
        }
    }

    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 10; j++) {
            data_in[i][j] = i * 11 + j;
        }
    }

    for(int i = 0; i < 20; i++) {
        for(int j = 0; j < 10; j++) {
            for(int n = 0; n < 4; n++) {
                data_out_ref[n][i] += data_in[n][j] * weights[i][j];
            }
        }
    }

    printf("REF:\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 20; j++) {
            printf("[%d][%d]: %d;  ", i, j, data_out_ref[i][j]);
        }
        printf("\n");
    }

    top(weights, data_in, data_out);

    printf("OUTPUT:\n");
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 20; j++) {
            printf("[%d][%d]: %d;  ", i, j, data_out[i][j]);
        }
        printf("\n");
    }

    test();

    return 0;

}


void test()
{
    int a[10][20];
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 20; j++) {
            a[i][j] = i * 10000 + j;
        }
    }


    int (*b)[20];

    b = &a[5];
    printf("%d\n", (*b)[0]);
    printf("%d\n", (*b)[1]);
    printf("%d\n", (*b)[2]);

    b = b + 1;
    printf("%d\n", (*b)[0]);
    printf("%d\n", (*b)[1]);
    printf("%d\n", (*b)[2]);
}


