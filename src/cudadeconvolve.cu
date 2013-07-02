/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

//#include <cuComplex.h>
#include <complex>
#include <iostream>
#include "utils.hpp"

using namespace std;

#define MAX_THREADS 1024


struct cuComplex
{
	float r;
	float i;

	__device__ cuComplex( float a, float b ) : r(a), i(b) {}

	__device__ float magnitude2( void )
	{
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}

	__device__ cuComplex operator+=(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}
};



__global__ void deconvolve(cuComplex *pn, cuComplex *data, 
	cuComplex *old_data, float *product, int pn_len)
{
	int threadsPerBlock = blockDim.x * blockDim.y;
	int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
	int threadId = threadIdx.x + (threadIdx.y * blockDim.x);
	int globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x);
	int n;
	int pn_idx;

	if(globalIdx >= pn_len) return;

	cuComplex s = cuComplex(0.0, 0.0);

	// TODO: this isn't the real deconvolve algo. PN loops back on itself here...
	for( n = 0; n < pn_len; n++){
		pn_idx = (globalIdx + n) % pn_len;
		s += data[n] * pn[pn_idx];
	}

	product[globalIdx] = s.magnitude2();
	//product[globalIdx] = 1; // ##### DEBUG OUTPUT - FIXME! ####### 

	/* old deconvolve ref.... 
	int i = threadIdx.x;
	int N = blockDim.x;
	int I = N - i;
	int n;
	cuComplex s = cuComplex(0.0, 0.0);

	for( n = 0; n < I; n++)
		s += data[n] * pn[n + i];
	for( n = i; n < N; n++)
		s += old_data[n] * pn[n + I];

	product[i] = s.magnitude2();
	*/
}

extern "C"
{	
	cuComplex *d_pcode;
	cuComplex *d_data1;
	cuComplex *d_data2;
	float *d_prod1;
	int h_buf_idx;
	size_t h_len;

	void init_deconvolve(complex<float> *h_pn, size_t len)
	{
		h_len = len;
		checkCudaErrors(cudaMalloc(&d_pcode, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_data1, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_data2, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_prod1, h_len * sizeof(float)));
		checkCudaErrors(cudaMemset(d_data2, 0, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMemcpy(d_pcode, h_pn, h_len, cudaMemcpyHostToDevice));

		// check for errors at end of init... 
		//  -- this is redundant with inline calls. 
		//checkCudaErrors(cudaGetLastError());
	}


	void start_deconvolve(complex<float> *h_data, float *h_product)
	{
		// Buffer switch
		if( 1 == h_buf_idx ) h_buf_idx = 0;
		else h_buf_idx = 1;

		// 
		cudaMemcpy(h_buf_idx?d_data1:d_data2, h_data, h_len, cudaMemcpyHostToDevice);

		// Define grids, blocks
		// issuing [MAX,1,1] grids
		// issuing [n,1,1] blocks
		// 
   		const dim3 blockSize(MAX_THREADS, 1, 1);
   		const int nBlocks = (h_len + MAX_THREADS - 1)/MAX_THREADS; // ceil of int division
		const dim3 gridSize(nBlocks, 1, 1); // Note: max 1024 threads/block! (r*c*d <=1024)
		
		// Task the SM's
		deconvolve<<<gridSize, blockSize>>>(d_pcode, h_buf_idx?d_data1:d_data2, h_buf_idx?d_data2:d_data1, d_prod1, h_len);
  		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
	    // Copy results to host
		cudaMemcpy(h_product, d_prod1, h_len * sizeof(float), cudaMemcpyDeviceToHost);
	}

}

//Free all the memory that we allocated
//TODO: check that this is comprehensive
void cleanup() {
  checkCudaErrors(cudaFree(d_pcode));
  checkCudaErrors(cudaFree(d_data1));
  checkCudaErrors(cudaFree(d_data2));
  checkCudaErrors(cudaFree(d_prod1));
}
