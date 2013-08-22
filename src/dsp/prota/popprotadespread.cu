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
#include "dsp/utils.hpp"

#include <boost/math/common_factor.hpp>

#include <cufft.h>

using namespace std;

#define MAX_THREADS_PER_BLOCK 1024
#define IFFT_PADDING_FACTOR 2


#define PN_MATCHED_FILTER_THREADS_PER_BLOCK 1000


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

	out[i] = in[i];

}

/**
 * pn_matched_filter
 * FIR filter for PN deconvolution. Uses maximum CUDA thread size.
 * Can be used as a partial too.
 * @param in double buffered input data (i.e. twice as long)
 * @param coef filter coefficients - time reversed PN spreading sequence
 * @param out output data
 */
__global__ void 
	__launch_bounds(PN_MATCHED_FILTER_THREADS_PER_BLOCK)
	pn_matched_filter(CuComplex *in, float *coef, CuComplex *out)
{
	int i = threadIdx.x;
	int I = blockDim.x;
	int n;
	cuComplex s;

	/// shared memory for PN code
	extern __shared__ float s_coef[];

	/// allocate vector array big enough to hold old and new data
	extern __shared__ float4 s_in[];

	/// recast into complex array
	cuComplex sp_in[] = (cuComplex*)s_in;

	// copy global memory to shared
	sp_in[i]     = in[i];     ///< double buffer
	sp_in[i + I] = in[i + I]; ///< new data
	s_coef[i]      = coef[i]; ///< fitler coefficients
	__syncthreads();

	// initialize accumulator
	s.x = 0.0;
	s.y = 0.0;

	for( n = 0; n < I; n++ ) 
		s += sp_in[i - n] * s_coef[n];

	out[i] = s;
}

extern "C"
{	
	cuComplex* d_dataa;
	cuComplex* d_datab;
	cuComplex* d_datac;
	cuComplex* d_datad;
	cuComplex* d_datad_upper;
	cuComplex* d_datae;
	float*     d_coef;
	float*     h_coef;
	cufftHandle plan1;
	size_t g_len_chan; ///< length of time series in samples
	size_t g_len_chan_padded; ///< length of interpolated time series
	size_t g_len_fft; ///< length of fft in samples
	size_t g_len_pn;
	size_t g_start_chan = 16059;
	size_t g_oversample_rate = 50;
	size_t g_len_pn_oversampled;

	const uint8_t pn_code_b[] = {
       0x67,0x7A,0xFA,0x1C,0x52,0x07,0x56,0x06,0x08,0x5C,0xBF,0xE4,0xE8,0xAE,0x88,0xDD,
       0x87,0xAA,0xAF,0x9B,0x04,0xCF,0x9A,0xA7,0xE1,0x94,0x8C,0x25,0xC0,0x2F,0xB8,0xA8,
       0xC0,0x1C,0x36,0xAE,0x4D,0x6E,0xBE,0x1F,0x99,0x0D,0x4F,0x86,0x9A,0x65,0xCD,0xEA,
       0x03,0xF0,0x92,0x52,0xDC,0x20,0x8E,0x69,0xFB,0x74,0xE6,0x13,0x2C,0xE7,0x7E,0x25,
       0xB5,0x78,0xFD,0xFE,0x33,0xAC,0x37,0x2E,0x6B,0x83,0xAC,0xB0,0x22,0x00,0x23,0x97,
       0xA6,0xEC,0x6F,0xB5,0xBF,0xFC,0xFD,0x4D,0xD4,0xCB,0xF5,0xED,0x1F,0x43,0xFE,0x58,
       0x23,0xEF,0x4E,0x82,0x32,0xD1,0x52,0xAF,0x0E,0x71,0x8C,0x97,0x05,0x9B,0xD9,0x82};

    /**
     * gpu_gen_pn_match_filter_coef
     * Generates the FIR filter coefficients for the PN matched filter.
     * @param pn_code spreading code
     * @param coef PN matched filter coefficients (len=size*oversample_factor)
     * @param size number of codes
     * @param oversample_factor number of times oversampled
     **/
    void gpu_gen_pn_match_filter_coef(uint8_t *pn_code, ///< in
    	                              float* coef,      ///< out
    	                              unsigned size,
    	                              unsigned oversample_factor)
    {
    	unsigned m, n, b, B;

    	for( n = 0; n < size; n++ )
    	{
    		B = n / 8;
    		b = n % 8;
    		for( m = 0; m < oversample_factor; m++ )
    			coef[n + m] = ((pn_code[size - B] >> b) & 0x1)?1.0f:-1.0f;
    	}
    }


	size_t gpu_channel_split(const complex<float> *h_data, complex<float> *out)
	{
		double ch_start, ch_end, ch_ctr;

		ch_start = 903626953 + (3200000 / (double)g_len_fft * (double)g_start_chan) - 1600000;
		ch_end = 903626953 + (3200000 / (double)g_len_fft * ((double)g_start_chan + 1040)) - 1600000;
		ch_ctr = (ch_start + ch_end) / 2.0;
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
		/*cudaMemcpy(d_datac,
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
  		checkCudaErrors(cudaGetLastError());*/

		// chop spectrum up into 50 spreading channels low side-band
		cudaMemcpy(d_datac,
			       d_datab + small_bin_start + small_bin_sideband,
			       small_bin_sideband * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);
		// chop spectrum up into 50 spreading channels high side-band
		cudaMemcpy(d_datac + g_len_fft - small_bin_sideband,
			       d_datab + small_bin_start,
			       small_bin_sideband * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);

		// swap double buffer
		cudaMemcpy(d_datad,
			       d_datad_upper,
			       g_len_chan * sizeof(cuComplex),
			       cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

  		// put back into time domain
		cufftExecC2C(plan1, (cufftComplex*)d_datac, (cufftComplex*)d_datad_upper, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

  		pn_matched_filter<<<1, PN_MATCHED_FILTER_THREADS_PER_BLOCK, PN_MATCHED_FILTER_THREADS_PER_BLOCK>>>(d_datad, d_coef, d_data_e);


  		// Copy results to host
		//cudaMemcpy(out, d_datad_upper, g_len_chan * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		
  		return 0;
	}

	size_t gpu_demod(complex<float> *out)
	{
		/*cuComplex* old_data = d_datad + g_len_chan_padded - h_decon_idx;
		cuComplex* new_data = old_data + g_len_pn;

  		// deconvolve PN codes
		//deconvolve<<<10, 80>>>(d_prncode, old_data, new_data, d_datae, g_len_pn);
		cudaThreadSynchronize();
		
		// Copy [deconvolved] results to host
		cudaMemcpy(out, d_datad, g_len_chan * sizeof(cuComplex), cudaMemcpyDeviceToHost);*/

		return 0;
	}

	void gpu_pn_matched_filter()
	{
		pn_matched_filter<<<1, PN_MATCHED_FILTER_THREADS_PER_BLOCK, PN_MATCHED_FILTER_THREADS_PER_BLOCK>>>();
	}


	void init_deconvolve(complex<float> *h_pn, size_t len_pn, size_t len_fft, size_t len_chan)
	{
		g_len_chan = len_chan;
		g_len_chan_padded = len_chan * IFFT_PADDING_FACTOR;
		g_len_fft = len_fft;
		g_len_pn = len_pn;
		g_len_pn_oversampled = g_len_pn * g_oversample_rate;

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_dataa, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datab, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, g_len_fft * sizeof(cuComplex)) * 2); // double buffered
		d_datad_upper = d_datad + g_len_chan;
		checkCudaErrors(cudaMalloc(&d_datae, g_len_pn_oversampled * sizeof(cuComplex)));

		checkCudaErrors(cudaMalloc(&d_coef,  g_len_pn_oversampled * sizeof(float)));
		h_coef = (float*)malloc(g_len_pn_oversampled * sizeof(float)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemset(d_dataa, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datab, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datac, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datad, 0, g_len_fft * sizeof(cuComplex)) * 2); // dobule buffered
		//checkCudaErrors(cudaMemset(d_datae, 0, g_len_pn_oversampled * sizeof(cuComplex)));

		gpu_gen_pn_match_filter_coef(pn_code_b, h_coef, g_len_pn, g_oversample_rate)
		checkCudaErrors(cudaMemcpy(d_coef, h_coef, g_len_pn_oversampled * sizeof(float)));
		free(h_coef);
		

	    // setup FFT plans
	    cufftPlan1d(&plan1, g_len_fft, CUFFT_C2C, 1);

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
	  checkCudaErrors(cudaFree(d_datae));
	  checkCudaErrors(cudaFree(d_coef));
	}

}