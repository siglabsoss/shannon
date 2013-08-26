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
#include <cstdio>
#include <cmath>
#include "dsp/utils.hpp"

#include <cufft.h>

using namespace std;

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

__global__ void rolling_dot_product(cuComplex *in, cuComplex *cfc, cuComplex *out, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int I = blockDim.x * gridDim.x;
	
	int F = (I / len);
	int f = (i / len) - (F / 2); // frequency
	int b = i % len; // fft bin

	out[i] = in[b] * cfc[(i+f)%len];
}

__global__ void peak_detection(cuComplex *in, float *peak, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int I = blockDim.x * gridDim.x;

	int F = (I / len);
	int f = (i / len) - (F / 2); // frequency
	int b = i % len; // fft bin
	float mag; // magnitude of peak
	unsigned si; // sortable integer

	// don't look for peaks in padding
	if( b < (len / 4) ) return;
	if( b >= (3 * len / 4) ) return;

	// take the magnitude of the detection
	mag = magnitude2(in[i]);

	// transform into sortable integer
	// https://devtalk.nvidia.com/default/topic/406770/cuda-programming-and-performance/atomicmax-for-float/
	si = *((unsigned*)&mag) ^ (-signed(*((unsigned*)&mag)>>31) | 0x80000000);

	// check to see if this is the highest recorded value
	atomicMax((unsigned*)peak, si);
}


extern "C"
{	
	cuComplex* d_dataa;
	cuComplex* d_datab;
	cuComplex* d_datac;
	cuComplex* d_datad;
	cuComplex* d_datad_upper;
	cufftHandle plan1;
	cufftHandle plan2;
	size_t g_len_chan; ///< length of time series in samples
	size_t g_len_chan_padded; ///< length of interpolated time series
	size_t g_len_fft; ///< length of fft in samples
	size_t g_start_chan = 16059;
	size_t g_oversample_rate = 50;


	size_t gpu_channel_split(const complex<float> *h_data, complex<float> *out)
	{
		//double ch_start, ch_end, ch_ctr;

/*		ch_start = 903626953 + (3200000 / (double)g_len_fft * (double)g_start_chan) - 1600000;
		ch_end = 903626953 + (3200000 / (double)g_len_fft * ((double)g_start_chan + 1040)) - 1600000;
		ch_ctr = (ch_start + ch_end) / 2.0;*/
		//printf("channel start: %f (%llu), end: %f, ctr: %f\r\n", ch_start, g_start_chan, ch_end, ch_ctr);

		// shift zero-frequency component to center of spectrum
		unsigned small_bin_start = (g_start_chan + (g_len_fft/2)) % g_len_fft;

		// calculate zero array size
		unsigned small_bin_padding = g_len_chan * (IFFT_PADDING_FACTOR-1);

		// calculate small bin side-band size
		unsigned small_bin_sideband = g_len_chan / 2;

		// copy new host data into device memory
		cudaMemcpy(d_dataa, h_data, g_len_fft * sizeof(cuComplex), cudaMemcpyHostToDevice);

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
			       g_len_chan_padded * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

		// put back into time domain
		cufftExecC2C(plan2, (cufftComplex*)d_datac, (cufftComplex*)d_datad_upper, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

  		// Copy results to host
		cudaMemcpy(out, d_datad_upper, g_len_chan * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		
  		return 0;
	}


	void init_deconvolve(size_t len_fft, size_t len_chan)
	{
		g_len_chan = len_chan;
		g_len_chan_padded = len_chan * IFFT_PADDING_FACTOR;
		g_len_fft = len_fft;

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_dataa, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datab, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, g_len_fft * sizeof(cuComplex) * 2)); // double buffered
		d_datad_upper = d_datad + g_len_chan;

		// initialize CUDA memory
		checkCudaErrors(cudaMemset(d_dataa, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datab, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datac, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datad, 0, g_len_fft * sizeof(cuComplex) * 2)); // dobule buffered

	    // setup FFT plans
	    cufftPlan1d(&plan1, g_len_fft, CUFFT_C2C, 1);
	    cufftPlan1d(&plan2, g_len_chan, CUFFT_C2C, 1);

	    printf("\n[Popwi::popprotadespread]: init deconvolve complete \n");
	}


	//Free all the memory that we allocated
	//TODO: check that this is comprehensive
	void cleanup() {
	  cufftDestroy(plan1);
	  checkCudaErrors(cudaFree(d_dataa));
	  checkCudaErrors(cudaFree(d_datab));
	  checkCudaErrors(cudaFree(d_datac));
	  checkCudaErrors(cudaFree(d_datad));
	}


	void gpu_rolling_dot_product(cuComplex *in, cuComplex *cfc, cuComplex *out, int len, int fbins)
	{
		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
		rolling_dot_product<<<fbins * 2, len / 2>>>(in, cfc, out, len);
		cudaThreadSynchronize();
	}

	void gpu_peak_detection(cuComplex* in, float* peak, int len, int fbins)
	{
		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
		peak_detection<<<fbins * 2, len / 2>>>(in, peak, len);
		cudaThreadSynchronize();
	}

}
