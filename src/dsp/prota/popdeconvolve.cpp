/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda/helper_cuda.h"

#include <core/popexception.hpp>

#include "popdeconvolve.hpp"

using namespace std;

namespace pop
{

#define SPREADING_LENGTH 512
#define SPREADING_BINS 80

extern "C" void gpu_rolling_dot_product(cuComplex *in, cuComplex *cfc, cuComplex *out, int len, int fbins);
extern "C" void gpu_peak_detection(cuComplex* in, float* peak, int len, int fbins);

PopProtADeconvolve::PopProtADeconvolve() : PopSink<complex<float> >( "PopProtADeconvolve", SPREADING_LENGTH ),
		PopSource<complex<float> >( "PopProtADeconvolve" )
{

}

PopProtADeconvolve::~PopProtADeconvolve()
{
	cufftDestroy(plan_fft);
	cufftDestroy(plan_deconvolve);
	checkCudaErrors(cudaFree(d_sts));
	checkCudaErrors(cudaFree(d_sfs));
	checkCudaErrors(cudaFree(d_cfc));
	checkCudaErrors(cudaFree(d_cfs));
	checkCudaErrors(cudaFree(d_cts));
	checkCudaErrors(cudaFree(d_peak));
}

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
void PopProtADeconvolve::gpu_gen_pn_match_filter_coef(
	const int8_t* prn, complex<float>* cfc,
	size_t  ncs, size_t osl, float bt)
{
	size_t n, m;
	float* p;  ///< phase
	float* fp; ///< filtered phase
	complex<float>* y;  ///< matched waveform
	complex<float>* yp; ///< interpolated matched waveform
	complex<float>* yc; ///< interpolated waveform conjugate
	complex<float>* yf; ///< fourier components
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
	yf = (complex<float>*) malloc( osl * sizeof(complex<float>) * 2 );

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
/*	for( m = 0; m < osl; m++ )
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
	}*/

	// complex conjugate and flip
	for( m = 0; m < osl; m++ )
	{
		yc[m].real(+y[osl-m].real());
		yc[m].imag(-y[osl-m].imag());
	}

	// pad and discrete fourier transform
	for( m = 0; m < 2 * osl; m++ )
	{
		yf[m] = complex<float>(0.0, 0.0);
		for( n = 0; n < osl; n++ )
		{
			q = (float)osl / 2.0 + (float)n; ///< padded index
			a = -2.0 * M_PI * (float)m * q / (2.0 * (float)osl);
			yf[m] += yc[n] * complex<float>( cos(a), sin(a) );
		}
	}

	// don't know why we need this step but this makes the output match matlab
	for( m = 0; m < 2 * osl; m++ )
	{
		cfc[m].real(-yf[m].imag());
		cfc[m].imag(yf[m].real());
	}

	// free all dynamically allocated memory
	free( p );
	free( fp );
	free( y );
	free( yp );
	free( yc );
	free( yf );
}
complex<float>* h_cfc;

void PopProtADeconvolve::init()
{
	

	// Init CUDA
 	int deviceCount = 0;
    cout << "initializing graphics card(s)...." << endl;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
        throw PopException("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    	throw PopException("There are no available device(s) that support CUDA\n");
    else
        cout << "Detected " << deviceCount << " CUDA Capable device(s)\n" << endl;

    // choose which device to use for this thread
    cudaSetDevice(0);

    // setup FFT plans
    cufftPlan1d(&plan_fft, SPREADING_LENGTH * 2, CUFFT_C2C, 1); // pad
    int rank_size = SPREADING_LENGTH * 2;
    cufftPlanMany(&plan_deconvolve, 1, &rank_size, 0, 1, 0, 0, 1, 0, CUFFT_C2C, SPREADING_BINS);

    // allocate device memory
    checkCudaErrors(cudaMalloc(&d_sts, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMalloc(&d_sfs, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMalloc(&d_cfc, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMalloc(&d_cfs, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMalloc(&d_cts, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMalloc(&d_peak, sizeof(float)));

    // initialize device memory
    checkCudaErrors(cudaMemset(d_sts, 0, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_sfs, 0, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_cfs, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_cts, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));

    // generate convolution filter coefficients
    h_cfc = (complex<float>*)malloc(2 * SPREADING_LENGTH * sizeof(complex<float>));
	gpu_gen_pn_match_filter_coef(pn, h_cfc, SPREADING_LENGTH, SPREADING_LENGTH /*oversampled*/, 0.5);
	checkCudaErrors(cudaMemcpy(d_cfc, h_cfc, 2 * SPREADING_LENGTH * sizeof(cuComplex), cudaMemcpyHostToDevice));
	//free(h_cfc);
}


unsigned IFloatFlip(unsigned f)
{
	unsigned mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}


void PopProtADeconvolve::process(const complex<float>* in, size_t len)
{
	unsigned n;
	float h_peak[10];

	//cout << "received " << len << " samples" << endl;

	if( len != SPREADING_LENGTH )
		throw PopException("size does not match filter");

	complex<float>* h_cts = get_buffer(len * SPREADING_BINS * 2);

	// copy new host data into device memory
	cudaMemcpy(d_sts, in - SPREADING_LENGTH, SPREADING_LENGTH * 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	// perform FFT on spectrum
	cufftExecC2C(plan_fft, d_sts, d_sfs, CUFFT_FORWARD);
	cudaThreadSynchronize();


	// rolling dot product
	gpu_rolling_dot_product(d_sfs, d_cfc, d_cfs, SPREADING_LENGTH * 2, SPREADING_BINS);
	cudaThreadSynchronize();


	// multiple ifft
	//for( n = 0; n < SPREADING_BINS; n++ )
	//{
	//	cufftExecC2C(plan_fft, (cufftComplex*)d_cfs + (n * SPREADING_LENGTH * 2), (cufftComplex*)d_cts + (n * SPREADING_LENGTH * 2), CUFFT_INVERSE);
	//	cudaThreadSynchronize();
	//}
	cufftExecC2C(plan_deconvolve, d_cfs, d_cts, CUFFT_INVERSE);
	cudaThreadSynchronize();
	cudaMemcpy(h_cts, d_cts, SPREADING_BINS * SPREADING_LENGTH * 2 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	// peak detection
	checkCudaErrors(cudaMemset(d_peak, 0, sizeof(float)));
	cudaThreadSynchronize();
	gpu_peak_detection(d_cts, d_peak, SPREADING_LENGTH * 2, SPREADING_BINS);
	cudaThreadSynchronize();
	cudaMemcpy(h_peak, d_peak, sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	// cast back to float from "sortable integer"
	unsigned a, b, c;
	//a = *((unsigned*)h_peak);
	//b = ((a >> 31) - 1) | 0x80000000;
	//c = a ^ b;
	float d;
	//d = *((float*)&c);
	(unsigned&)d = IFloatFlip((unsigned&)h_peak);




	if( d > 10.5e9 )
		cout << "peak: " << d << endl;


	//PopSource<complex<float> >::process(h_cfc, 1024);
	PopSource<complex<float> >::process();
}

} // namespace pop
