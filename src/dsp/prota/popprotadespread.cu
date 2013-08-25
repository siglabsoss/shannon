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

extern "C"
{	
	cuComplex* d_dataa;
	cuComplex* d_datab;
	cuComplex* d_datac;
	cuComplex* d_datad;
	cuComplex* d_datad_upper;
	cuComplex* d_cfc;
	cufftHandle plan1;
	cufftHandle plan2;
	size_t g_len_chan; ///< length of time series in samples
	size_t g_len_chan_padded; ///< length of interpolated time series
	size_t g_len_fft; ///< length of fft in samples
	size_t g_len_pn;
	size_t g_start_chan = 16059;
	size_t g_oversample_rate = 50;
	size_t g_len_pn_oversampled;
	/// spreading code m4k_001
	const int8_t pn[] = {-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1,
				       1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
				      -1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
				      -1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,
				      -1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1,
				       1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,
				       1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1,
				       1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,
				       1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,
				       1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,
				      -1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,
				       1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,
				       1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1,
				       1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1,
				       1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,
				       1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,
				       1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
				      -1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1,
				      -1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,
				       1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
				       1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,
				      -1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,
				       1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,
				       1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,
				      -1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,
				       1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,
				       1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
				       1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,
				       1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1,
				       1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,
				      -1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,
				      -1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1};


    /**
     * gpu_gen_pn_match_filter_coef
     * Generates the convolution filter coefficients for a given PN code.
     * The cfc is padded in the time domain to allow for discrete convolutions.
     * @param prn spreading code in format [-1, 1, 1,-1, ...]
     * @param cfc convolution filter coefficients out (must be 2*osl in length)
     * @param ncs number of chips per symbol
     * @param osl oversampled symbol length (should be >= prn length)
     * @param bt Gaussian filter value. -3dB bandwidth symbol time.
     **/
    void gpu_gen_pn_match_filter_coef(const int8_t* prn, complex<float>* cfc,
    	                              size_t  ncs, size_t osl, float bt)
    {
    	size_t n, m;
    	float* p;  ///< phase
    	float* fp; ///< filtered phase
    	complex<float>* y;  ///< matched waveform
    	complex<float>* yp; ///< interpolated matched waveform
    	complex<float>* yc; ///< interpolated waveform conjugate
    	float  t[3] = {-1, 0, 1}; ///< gaussian points
    	float  h[3]; ///< gaussian filter
    	float  alpha; ///< gaussian alpha
    	float  a, b, q;

    	// allocate buffers
    	p = (float*) malloc( ncs * sizeof(float) );
    	fp = (float*) malloc( ncs * sizeof(float) );
    	y = (complex<float>*) malloc( ncs * sizeof(complex<float>) );
    	yp = (complex<float>*) malloc( osl * sizeof(complex<float>) );
    	yc = (complex<float>*) malloc( osl * sizeof(complex<float>) );

    	p[0] = 0; ///< starting phase

    	// generate phase map
    	for( n = 1; n < ncs; n++ )
    	{
    		p[n] = p[n-1] + (float)prn[n] * M_PI / 2;
    	}

    	// gaussian filter
    	alpha = sqrt( log(2.0) / 2.0 ) / (double)bt;
    	a = 0.0;
    	for( n = 0; n < 3; n++ )
    	{
    		b = t[n] * M_PI / alpha;
    		h[n] = sqrt(M_PI) / alpha * exp( -(b * b) );
    		a += h[n];
    	}

    	// normalize
    	for( n = 0; n < 3; n++ )
    	{
    		h[n] /= a;
    	}

    	// filter (convolve) phase map
    	fp[0] = p[0] * h[1] + p[1] * h[2];
    	for( n = 1; n < ncs - 1; n++ )
    	{
    		fp[n] = p[n-1] * h[0] + p[n] * h[1] + p[n+1] * h[2];
    	}
    	fp[ncs] = p[ncs-1] * h[0] + p[ncs] * h[1];

    	// generate
    	for( n = 0; n < ncs; n++ )
    	{
    		y[n].real(cos(fp[n]));
    		y[n].imag(sin(fp[n]));
    	}

    	// sinc interpolate to sample frequency
    	for( m = 0; m < osl; m++ )
    	{
    		yp[m] = complex<float>(0.0, 0.0);
    		for( n = 0; n < ncs; n++ )
    		{
    			a = M_PI * ( (float)m / (float)osl * (float)ncs - (float)n );
    			if( 0 == a )
    				yp[m] += y[n];
    			else
    				yp[m] += sin(a) / a * y[n];
    		}
    	}

    	// complex conjugate and flip
    	for( m = 0; m < osl; m++ )
    	{
    		yc[m].real(+yp[osl-m].real());
    		yc[m].imag(-yp[osl-m].imag());
    	}

    	// pad and discrete fourier transform
    	for( m = 0; m < 2 * osl; m++ )
    	{
    		cfc[m] = complex<float>(0.0, 0.0);
    		for( n = 0; n < osl; n++ )
    		{
    			q = (float)osl / 4.0 + (float)n; ///< padded index
    			a = -2.0 * M_PI * (float)m * q / (2.0 * (float)osl);
    			cfc[m] += yc[n] * complex<float>( cos(a), sin(a) );
    		}
    	}

    	// free all dynamically allocated memory
    	free( p );
    	free( fp );
    	free( y );
    	free( yp );
    	free( yc );
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


	void init_deconvolve(complex<float> *h_pn, size_t len_pn, size_t len_fft, size_t len_chan)
	{
		complex<float>* h_cfc;

		g_len_chan = len_chan;
		g_len_chan_padded = len_chan * IFFT_PADDING_FACTOR;
		g_len_fft = len_fft;
		g_len_pn = len_pn;
		g_len_pn_oversampled = g_len_pn * g_oversample_rate;

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_dataa, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datab, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, g_len_fft * sizeof(cuComplex) * 2)); // double buffered
		checkCudaErrors(cudaMalloc(&d_cfc,   g_len_pn  * sizeof(cuComplex) * 2)); // padded
		d_datad_upper = d_datad + g_len_chan;

		// initialize CUDA memory
		checkCudaErrors(cudaMemset(d_dataa, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datab, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datac, 0, g_len_fft * sizeof(cuComplex)));
		checkCudaErrors(cudaMemset(d_datad, 0, g_len_fft * sizeof(cuComplex) * 2)); // dobule buffered

	    // setup FFT plans
	    cufftPlan1d(&plan1, g_len_fft, CUFFT_C2C, 1);
	    cufftPlan1d(&plan2, g_len_chan, CUFFT_C2C, 1);

	    // generate convolution filter coefficients
	    h_cfc = (complex<float>*)malloc(2 * g_len_pn * sizeof(complex<float>));
		gpu_gen_pn_match_filter_coef(pn, h_cfc, 512, 512 /*oversampled*/, 0.5);
		checkCudaErrors(cudaMemcpy(d_cfc, h_cfc, 2 * g_len_pn * sizeof(cuComplex), cudaMemcpyHostToDevice));
		free(h_cfc);

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
	  checkCudaErrors(cudaFree(d_cfc));
	}

}
