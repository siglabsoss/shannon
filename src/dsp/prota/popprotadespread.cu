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
#include "dsp/utils.hpp"

#include <cufft.h>

using namespace std;

#define MAX_THREADS_PER_BLOCK 1024


__device__ float magnitude2( cuComplex& in )
{
	return in.x * in.x + in.y * in.y;
}

__device__ cuComplex operator*(const cuComplex& a, const cuComplex& b)
{
	cuComplex r;
	r.x = b.x*a.x - b.y*a.y;
	r.y = b.y*a.x + b.x*a.y;
	return r;
}

__device__ cuComplex operator+=(const cuComplex& a, const cuComplex& b)
{
	cuComplex r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	return r;
}


__global__ void deconvolve(cuComplex *pn, cuComplex *data,
	float *product, int pn_len)
{
	int n;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int temp;

	// move data to local memory
	// Moving in chunks of 1024 samples(4kB). SM = 48kB
	__shared__ cuComplex smem_data[1024];
	__shared__ cuComplex smem_pn[1024]; //TODO: iterate over blocks

	smem_data[threadIdx.x] = data[threadIdx.x];
	smem_pn[threadIdx.x] = pn[threadIdx.x];

	// Must sync to ensure all data copied in
	__syncthreads();

	//cuComplex s = cuComplex(0.0, 0.0);
	cuComplex s;
	s.x = 0.0;
	s.y = 0.0;

	// Perform deconvolutoin
	for( n = 0; n < pn_len; n++)
		temp = n % 1024;
		//s += shrd_data[n + i] * shrd_pn[n];
		s += smem_data[temp] * smem_pn[temp]; // Indexing all wrong here. Computation speed test only


	product[i] = magnitude2(s);
}

extern "C"
{	
	cuComplex *d_prncode;
	cuComplex *d_dataold;
	cuComplex *d_dataa;
	cuComplex *d_datab;
	cuComplex *d_datac;
	cuComplex *d_datad;
	cufftHandle plan1;
	cufftHandle plan2;
	float *d_product;
	size_t h_len; ///< length of data in samples


	void start_deconvolve(const complex<float> *h_data, complex<float> *h_product)
	{
		unsigned small_bin_start;
		unsigned small_bin_width = 1040;

		// copy new memory to old
		cudaMemcpy(d_dataold, d_dataa, h_len * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// copy new host data into device memory
		cudaMemcpy(d_dataa, h_data, h_len * sizeof(cuComplex), cudaMemcpyHostToDevice);

		// Task the SM's
		//deconvolve<<<64, 1024>>>(d_prncode, d_dataold, d_product, h_len);

		// perform FFT on spectrum
		cufftExecC2C(plan1, (cufftComplex*)d_dataa, (cufftComplex*)d_datab, CUFFT_FORWARD);
		cudaThreadSynchronize();

		// shift zero-frequency component to center of spectrum
		small_bin_start = ((16059 + 32768) % 65536);
		// chop spectrum up into 50 spreading channels
		cudaMemcpy(d_datac, d_datab + small_bin_start, 1040 * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// put back into time domain
		cufftExecC2C(plan2, (cufftComplex*)d_datac, (cufftComplex*)d_datad, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());
		
	    // Copy results to host
		cudaMemcpy(h_product, d_datad, small_bin_width * sizeof(complex<float>), cudaMemcpyDeviceToHost);
		//memcpy(h_product, h_data, 1040 * sizeof(complex<float>));
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
		d_dataa = d_dataold + h_len; ///< make this sequential to old data
		checkCudaErrors(cudaMalloc(&d_product, h_len * sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_datab, 655536 * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, 1040 * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, 1040 * sizeof(cuComplex)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemcpy(d_prncode, h_pn, h_len, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_dataold, 0, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_dataa, 0, h_len * sizeof(cuComplex)));		
		checkCudaErrors(cudaMemset(d_product, 0, h_len * sizeof(float)));

	    // setup FFT plans
	    cufftPlan1d(&plan1, 65536, CUFFT_C2C, 1);
	    cufftPlan1d(&plan2, 1040, CUFFT_C2C, 1);
	}

	//Free all the memory that we allocated
	//TODO: check that this is comprehensive
	void cleanup() {
	  cufftDestroy(plan1);
	  cufftDestroy(plan2);
	  checkCudaErrors(cudaFree(d_prncode));
	  checkCudaErrors(cudaFree(d_dataold));
	  checkCudaErrors(cudaFree(d_dataa));
	  checkCudaErrors(cudaFree(d_datab));
	  checkCudaErrors(cudaFree(d_datac));
	  checkCudaErrors(cudaFree(d_datad));
	  checkCudaErrors(cudaFree(d_product));
	}

}