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
#define IFFT_PADDING_FACTOR 2


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

__device__ cuComplex operator+(const cuComplex& a, const cuComplex& b)
{
	cuComplex r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	return r;
}


__global__ void deconvolve(cuComplex *pn, cuComplex *data,
	float *product, int pn_len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int memIdx;

	if(i >= pn_len) return;

	// move data to local memory
	// Shared mem size = 48kB
	//__shared__ cuComplex smem_data[2080*2];
	__shared__ cuComplex smem_pn[2080];

	// Copy in contiguous chunks of data into SMEM. __sync after each chunk to ensure coalesced access
	for(int memRow = 0; memRow < gridDim.x; memRow++){
		memIdx = memRow * blockDim.x + threadIdx.x;
		if(memIdx < pn_len){
			//smem_data[memIdx] = data[memIdx];
			smem_pn[memIdx] = pn[memIdx];
		}
		__syncthreads();
	}
	// Must sync to ensure all data copied in

	cuComplex s;
	s.x = 0.0;
	s.y = 0.0;
	// Perform deconvolutoin
	for(int n = 0; n < pn_len; n++){
		s = s + (data[n+i] * smem_pn[n]);
	}
	
	// output mag result
	product[i] = magnitude2(s);
}

extern "C"
{	
	cuComplex *d_prncode;
	cuComplex *d_dataold;
	cuComplex *d_dataa;
	cuComplex *d_datab;
	cuComplex *d_datac;
	cuComplex *d_datac_padded;
	cuComplex *d_datad;
	cufftHandle plan1;
	cufftHandle plan2;
	float *d_product;
	size_t h_len; ///< length of data in samples


	void start_deconvolve(const complex<float> *h_data, complex<float> *h_product)
	{
		unsigned small_bin_start;
		unsigned small_bin_width = 1040;
		unsigned small_bin_width_padded = small_bin_width * IFFT_PADDING_FACTOR;


		// copy new memory to old
		cudaMemcpy(d_dataold, d_dataa, h_len * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// copy new host data into device memory
		cudaMemcpy(d_dataa, h_data, h_len * sizeof(cuComplex), cudaMemcpyHostToDevice);

		// perform FFT on spectrum
		cufftExecC2C(plan1, (cufftComplex*)d_dataa, (cufftComplex*)d_datab, CUFFT_FORWARD);
		cudaThreadSynchronize();

		// shift zero-frequency component to center of spectrum
		small_bin_start = ((16059 + 32768) % 65536);
		// chop spectrum up into 50 spreading channels
		//cudaMemcpy(d_datac, d_datab + small_bin_start, 1040 * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
		// >> cpy into longer, padded vector
		cudaMemcpy(d_datac_padded, d_datab + small_bin_start, 1040 * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

		// put back into time domain
		cufftExecC2C(plan2, (cufftComplex*)d_datac_padded, (cufftComplex*)d_datad, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());
		
	    // Copy [IFFT] results to host
		//cudaMemcpy(h_product, d_datad, small_bin_width_padded * sizeof(complex<float>), cudaMemcpyDeviceToHost);

		// Task the SM's
		// 1040 * 2 = 2080 samples
		// -> 128 Th/Bl
		// -> ~17 Bl
		deconvolve<<<17, 128>>>(d_prncode, d_dataold, d_product, small_bin_width_padded);
		
		// Copy [deconvolved] results to host
		cudaMemcpy(h_product, d_product, small_bin_width_padded * sizeof(float), cudaMemcpyDeviceToHost);
	}


	void init_deconvolve(complex<float> *h_pn, size_t len)
	{
		h_len = len;

		// verify that we're a multiple of samples of a thread block
		//if( 0 != (h_len % MAX_THREADS_PER_BLOCK) )
		//	throw runtime_error("[POPGPU] - sample length needs to be multiple of block size.\r\n");

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_prncode, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_dataold, h_len * sizeof(cuComplex) * 2));
		d_dataa = d_dataold + h_len; ///< make this sequential to old data
		checkCudaErrors(cudaMalloc(&d_product, h_len * sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_datab, 655536 * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, 1040 * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac_padded, 1040 * IFFT_PADDING_FACTOR * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, 1040 * IFFT_PADDING_FACTOR * sizeof(cuComplex)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemcpy(d_prncode, h_pn, h_len, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_dataold, 0, h_len * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_dataa, 0, h_len * sizeof(cuComplex)));		
		checkCudaErrors(cudaMemset(d_product, 0, h_len * sizeof(float)));
		checkCudaErrors(cudaMemset(d_datac_padded, 0, 1040 * IFFT_PADDING_FACTOR * sizeof(cuComplex)));

	    // setup FFT plans
	    cufftPlan1d(&plan1, 65536, CUFFT_C2C, 1);
	    cufftPlan1d(&plan2, 1040 * IFFT_PADDING_FACTOR, CUFFT_C2C, 1);
	    printf("[Popwi::popprotadespread]: init deconvolve complete \n");
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