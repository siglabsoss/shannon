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



__global__ void deconvolve(cuComplex *pn, cuComplex *data,
	float *product, int pn_len)
{
	int n;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	cuComplex s = cuComplex(0.0, 0.0);

	for( n = 0; n < pn_len; n++)
		s += data[n + i] * pn[n];

	product[i] = s.magnitude2();
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
		// copy new memory to old
		cudaMemcpy(d_dataold, d_datanew, h_len * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// copy new host data into device memory
		cudaMemcpy(d_datanew, h_data, h_len * sizeof(cuComplex), cudaMemcpyHostToDevice);

		// Task the SM's
		deconvolve<<<64, 1024>>>(d_prncode, d_dataold, d_product, h_len);
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
		checkCudaErrors(cudaMalloc(&d_dataold, h_len * sizeof(cuComplex) * 2));
		d_datanew = d_dataold + h_len; ///< make this sequential to old data
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