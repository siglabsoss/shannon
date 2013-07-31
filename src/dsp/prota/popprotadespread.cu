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

#include <boost/math/common_factor.hpp>

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

#define PN_LEN 800
#define SHARED_MEMORY_STEPS 2


__global__ void deconvolve(cuComplex *pn, cuComplex *old, cuComplex *in, cuComplex *out, int pn_len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int memIdx1;
	int n;

	cuComplex s;
	
	// shared memory size 48kB
	__shared__ cuComplex smem_data[PN_LEN * 2]; // 12,800 bytes
	cuComplex* smem_new_data_ptr = smem_data + PN_LEN;
	__shared__ cuComplex smem_pn[PN_LEN]; // 6,400 bytes = 19,200 total

	s.x = 0.0;
	s.y = 0.0;

	if(i >= pn_len) return;

	// copy old buffer into shared memory
	for( n = 0; n < gridDim.x; n++ )
	{
		memIdx1 = n * blockDim.x + threadIdx.x;
		smem_data[memIdx1] = old[memIdx1];
	}

	// copy new buffer into shared memory
	for( n = 0; n < gridDim.x; n++ )
	{
		memIdx1 = n * blockDim.x + threadIdx.x;
		smem_new_data_ptr[memIdx1] = in[memIdx1];
	}

	// copy PN code into shared memory
	for( n = 0; n < gridDim.x; n++ )
	{
		memIdx1 = n * blockDim.x + threadIdx.x;
		smem_pn[memIdx1] = pn[memIdx1];
	}

	// Must sync to ensure all data copied in
	__syncthreads();

	// // Perform deconvolutoin
	for(n = 0; n < pn_len; n++)
	{
	 	s = smem_data[n + i] * smem_pn[n] + s;
	}

	out[i] = s;

}

extern "C"
{	
	cuComplex *d_prncode;
	cuComplex *d_dataa;
	cuComplex *d_datab;
	cuComplex *d_datac;
	cuComplex *d_datad;
	cuComplex *d_datad_upper;
	cuComplex *d_datae;
	cufftHandle plan1;
	cufftHandle plan2;
	size_t h_len_chan; ///< length of time series in samples
	size_t h_len_chan_padded; ///< length of interpolated time series
	size_t h_len_fft; ///< length of fft in samples
	size_t h_len_pn;
	size_t h_decon_idx; ///< index of deconvolution operation


	size_t gpu_channel_split(const complex<float> *h_data)
	{
		// shift zero-frequency component to center of spectrum
		unsigned small_bin_start = (16059 + (h_len_fft/2)) % h_len_fft;;

		// calculate zero array size
		unsigned small_bin_padding = h_len_chan * (IFFT_PADDING_FACTOR-1);

		// calculate small bin side-band size
		unsigned small_bin_sideband = h_len_chan / 2;

		// copy new host data into device memory
		cudaMemcpy(d_dataa, h_data, h_len_fft * sizeof(cuComplex), cudaMemcpyHostToDevice);

		// perform FFT on spectrum
		cufftExecC2C(plan1, (cufftComplex*)d_dataa, (cufftComplex*)d_datab, CUFFT_FORWARD);
		cudaThreadSynchronize();

		
		// chop spectrum up into 50 spreading channels low side-band
		cudaMemcpy(d_datac,
			       d_datab + small_bin_start + small_bin_sideband,
			       small_bin_sideband * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);
		// chop spectrum up into 50 spreading channels high side-band
		cudaMemcpy(d_datac + small_bin_sideband + small_bin_padding,
			       d_datab + small_bin_start,
			       small_bin_sideband * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);

		// swap double buffer
		cudaMemcpy(d_datad,
			       d_datad_upper,
			       h_len_chan_padded * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

		// put back into time domain
		cufftExecC2C(plan2, (cufftComplex*)d_datac, (cufftComplex*)d_datad_upper, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());
		
  		h_decon_idx += h_len_chan_padded;
  		return h_decon_idx;
	}

	size_t gpu_demod(complex<float> *out)
	{
		cuComplex* old_data = d_datad + h_len_chan_padded - h_decon_idx;
		cuComplex* new_data = old_data + h_len_pn;

  		// deconvolve PN codes
		deconvolve<<<1, 800>>>(d_prncode, old_data, new_data, d_datae, h_len_pn);
		cudaThreadSynchronize();
		
		// Copy [deconvolved] results to host
		cudaMemcpy(out, d_datae, h_len_pn * sizeof(cuComplex), cudaMemcpyDeviceToHost);

		h_decon_idx -= h_len_pn;
		return h_decon_idx;
	}


	void init_deconvolve(complex<float> *h_pn, size_t len_pn, size_t len_fft, size_t len_chan)
	{
		h_len_chan = len_chan;
		h_len_chan_padded = len_chan * IFFT_PADDING_FACTOR;
		h_len_fft = len_fft;
		h_len_pn = len_pn;
		h_decon_idx = 0;

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_prncode, h_len_pn * sizeof(cuComplex)));

		checkCudaErrors(cudaMalloc(&d_dataa, h_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datab, h_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, h_len_chan_padded * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, 2 * h_len_chan_padded * sizeof(cuComplex))); // double buffered
		d_datad_upper = d_datad + h_len_chan_padded;
		checkCudaErrors(cudaMalloc(&d_datae, h_len_pn * sizeof(cuComplex)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemcpy(d_prncode, h_pn, h_len_pn * sizeof(cuComplex), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemset(d_dataa, 0, h_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datab, 0, h_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datac, 0, h_len_chan_padded * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datad, 0, 2 * h_len_chan_padded * sizeof(cuComplex))); // dobule buffered
		checkCudaErrors(cudaMemset(d_datae, 0, h_len_pn * sizeof(cuComplex)));
		

	    // setup FFT plans
	    cufftPlan1d(&plan1, h_len_fft, CUFFT_C2C, 1);
	    cufftPlan1d(&plan2, h_len_chan_padded, CUFFT_C2C, 1);

	    printf("\n[Popwi::popprotadespread]: init deconvolve complete \n");
	}


	//Free all the memory that we allocated
	//TODO: check that this is comprehensive
	void cleanup() {
	  cufftDestroy(plan1);
	  cufftDestroy(plan2);
	  checkCudaErrors(cudaFree(d_prncode));
	  checkCudaErrors(cudaFree(d_dataa));
	  checkCudaErrors(cudaFree(d_datab));
	  checkCudaErrors(cudaFree(d_datac));
	  checkCudaErrors(cudaFree(d_datad));
	  checkCudaErrors(cudaFree(d_datae));
	}

}