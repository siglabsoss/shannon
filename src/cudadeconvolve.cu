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
#include <stdexcept>
#include "utils.hpp"

using namespace std;

#define MAX_THREADS_PER_BLOCK 1024


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



__global__ void deconvolve(cuComplex *pn, cuComplex *datanew, 
	cuComplex *dataold, float *product, int pn_len)
{
	/*int threadsPerBlock = blockDim.x * blockDim.y;
	int blockId = blockIdx.x + (blockIdx.y * gridDim.x);
	int threadId = threadIdx.x + (threadIdx.y * blockDim.x);
	int globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x);
	int n;
	int pn_idx;*/

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//if(globalIdx >= pn_len) return;

	//cuComplex s = cuComplex(0.0, 0.0);

	product[i] = (float)i;

	//product[0] = 3.14f;

	// TODO: this isn't the real deconvolve algo. PN loops back on itself here...
	/*for( n = 0; n < pn_len; n++){
		pn_idx = (globalIdx + n) % pn_len;
		s += data[n] * pn[pn_idx];
	}*/

	//product[globalIdx] = s.magnitude2();
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
	cuComplex *d_prncode;
	cuComplex *d_dataold;
	cuComplex *d_datanew;
	float *d_product;
	size_t h_len; ///< length of data in samples


	void start_deconvolve(complex<float> *h_data, float *h_product)
	{
		/*for(unsigned n = 0; n < 65535; n++)
			h_product[n] = (float)n;*/

		// copy new memory to old
		cudaMemcpy(d_dataold, d_datanew, h_len * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// copy new host data into device memory
		cudaMemcpy(d_datanew, h_data, h_len * sizeof(cuComplex), cudaMemcpyHostToDevice);

		// Task the SM's
		deconvolve<<<64, 1024>>>(d_prncode, d_datanew, d_dataold, d_product, h_len);
  		checkCudaErrors(cudaGetLastError());
		
	    // Copy results to host
		cudaMemcpy(h_product, d_product, h_len * sizeof(float), cudaMemcpyDeviceToHost);
	}


	void init_deconvolve(complex<float> *h_pn, size_t len)
	{
		h_len = len;

		// verify that we're a multiple of samples of a thread block
		if( 0 != (h_len % MAX_THREADS_PER_BLOCK) )
			throw runtime_error("[POPGPU] - sample length needs to be multiple of block size.\r\n");

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_prncode, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_dataold, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datanew, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_product, h_len * sizeof(float)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemcpy(d_prncode, h_pn, h_len, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_dataold, 0, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datanew, 0, h_len * sizeof(cuComplex)));		
		checkCudaErrors(cudaMemset(d_product, 0, h_len * sizeof(float)));
	}

	//Free all the memory that we allocated
	//TODO: check that this is comprehensive
	void cleanup() {
	  checkCudaErrors(cudaFree(d_prncode));
	  checkCudaErrors(cudaFree(d_dataold));
	  checkCudaErrors(cudaFree(d_datanew));
	  checkCudaErrors(cudaFree(d_product));
	}

}