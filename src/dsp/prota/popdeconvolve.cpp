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
#include <complex>
#include <algorithm>    // std::min
#include "boost/tuple/tuple.hpp"

#include "cuda/helper_cuda.h"

#include <core/popexception.hpp>

#include <dsp/prota/popchanfilter.cuh>
#include "popdeconvolve.hpp"

//#define DEBUG_POPDECONVOLVE_TIME

using namespace std;

namespace pop
{

#define SPREADING_LENGTH 4096
#define SPREADING_BINS 400
#define MAX_SIGNALS_PER_SPREAD (32) // how much memory to allocate for detecting signal peaks

extern "C" void gpu_rolling_dot_product(cuComplex *in, cuComplex *cfc, cuComplex *out, int len, int fbins);
extern "C" void gpu_peak_detection(cuComplex* in, float* peak, int len, int fbins);
extern "C" void gpu_threshold_detection(cuComplex* d_in, int* d_out, unsigned int *d_outLen, int outLenMax, float threshold, int len, int fbins);
extern "C" void thrust_peak_detection(cuComplex* in, thrust::device_vector<float>* d_mag_vec, float* peak, int* index, int len, int fbins);
extern "C" void init_popdeconvolve(thrust::device_vector<float>** d_mag_vec, size_t size);


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
	const int8_t m4k_001[] = {-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1};


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
	double* p;  ///< phase
	double* fp; ///< filtered phase
	complex<double>* y;  ///< matched waveform
	complex<double>* yp; ///< interpolated matched waveform
	complex<double>* yc; ///< interpolated waveform conjugate
	complex<double>* yf; ///< fourier components
	double  t[3] = {-1, 0, 1}; ///< gaussian points
	double  h[3]; ///< gaussian filter
	double  alpha; ///< gaussian alpha
	double  a, b, q;

	// allocate buffers
	p = (double*) malloc( ncs * sizeof(double) );
	fp = (double*) malloc( ncs * sizeof(double) );
	y = (complex<double>*) malloc( ncs * sizeof(complex<double>) );
	yp = (complex<double>*) malloc( osl * sizeof(complex<double>) );
	yc = (complex<double>*) malloc( osl * sizeof(complex<double>) );
	yf = (complex<double>*) malloc( osl * sizeof(complex<double>) * 2 );

	p[0] = 0; ///< starting phase

	// generate phase map
	for( n = 1; n < ncs; n++ )
	{
		p[n] = p[n-1] + (double)prn[n] * M_PI / 2;
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
//	for( m = 0; m < osl; m++ )
//	{
//		yp[m] = complex<double>(0.0, 0.0);
//		for( n = 0; n < ncs; n++ )
//		{
//			a = M_PI * ( (double)m / (double)osl * (double)ncs - (double)n );
//			if( 0 == a )
//				yp[m] += y[n];
//			else
//				yp[m] += sin(a) / a * y[n];
//		}
//	}

	// complex conjugate and flip
	for( m = 0; m < osl; m++ )
	{
		yc[m].real(+y[osl-m].real());
		yc[m].imag(-y[osl-m].imag());
	}

	// pad and discrete fourier transform
	for( m = 0; m < 2 * osl; m++ )
	{
		yf[m] = complex<double>(0.0, 0.0);
		for( n = 0; n < osl; n++ )
		{
			q = (double)osl / 2.0 + (double)n; ///< padded index
			a = -2.0 * M_PI * (double)m * q / (2.0 * (double)osl);
			yf[m] += yc[n] * complex<double>( cos(a), sin(a) );
		}
		yf[m] /= 2 * osl;
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
    cudaSetDevice(1);

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
    checkCudaErrors(cudaMalloc(&d_peaks, MAX_SIGNALS_PER_SPREAD * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_peaks_len, sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&d_peak, sizeof(float)));

    // allocate thrust memory
    init_popdeconvolve(&d_mag_vec, SPREADING_LENGTH * SPREADING_BINS * 2);



    // initialize device memory
    checkCudaErrors(cudaMemset(d_sts, 0, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_sfs, 0, SPREADING_LENGTH * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_cfs, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_cts, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(cuComplex)));
    checkCudaErrors(cudaMemset(d_peaks, 0, MAX_SIGNALS_PER_SPREAD * sizeof(int)));

    // generate convolution filter coefficients
    cout << "generating spreading codes..." << endl;
    h_cfc = (complex<float>*)malloc(2 * SPREADING_LENGTH * sizeof(complex<float>));
	gpu_gen_pn_match_filter_coef(m4k_001, h_cfc, SPREADING_LENGTH, SPREADING_LENGTH /*oversampled*/, 0.5);
	checkCudaErrors(cudaMemcpy(d_cfc, h_cfc, 2 * SPREADING_LENGTH * sizeof(cuComplex), cudaMemcpyHostToDevice));
	cout << "done generating spreading codes" << endl;
	//free(h_cfc);
}


unsigned IFloatFlip(unsigned f)
{
	unsigned mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}


float magnitude2( const complex<float>& in )
{
	return in.real() * in.real() + in.imag() * in.imag();
}

// we need to check the points forwards and backwards in time, we also need to check up and down in fbins
// then we also check the corners
bool sampleIsLocalMaxima(const complex<float>* data, int sample, int spreadLength, int fbins)
{
	//TODO: we don't have a circular buffer of the final data, so if a sample lies on a boundary
	// we can't check for local maxima by negativly indexing to verify local maximums

	int sampleTime, sampleBin;

	float surroundingMax = 0.0; // maximum of surrounding points

	vector<int> check; // vector of indices we want to check

	check.push_back( sample+1 ); // right
	check.push_back( sample-1 ); // left
	check.push_back( sample - spreadLength ); // up
	check.push_back( sample + spreadLength ); // down
	check.push_back( sample - spreadLength - 1 ); // up left
	check.push_back( sample - spreadLength + 1 ); // up right
	check.push_back( sample + spreadLength - 1 ); // down left
	check.push_back( sample + spreadLength + 1 ); // down right

	for( unsigned i = 0; i < check.size(); i++ )
	{
		// bound to edges of array
		int testSample = min(max(check[i],0), spreadLength*fbins);

		if( testSample != check[i] )
			cout << "hit boundary in sampleIsLocalMaxima()" << endl;

		// compute maximum of surrounding indices
		surroundingMax = max(surroundingMax, magnitude2(data[testSample]));
	}

	// is local maxima?
	return ( magnitude2(data[sample]) > surroundingMax );
}


void PopProtADeconvolve::process(const complex<float>* in, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size)
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


	// perform IFFT on dot product
	cufftExecC2C(plan_deconvolve, d_cfs, d_cts, CUFFT_INVERSE);
	cudaThreadSynchronize();
	cudaMemcpy(h_cts, d_cts, SPREADING_BINS * SPREADING_LENGTH * 2 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	float threshold = 3.7e7;

	// threshold detection
	gpu_threshold_detection(d_cts, d_peaks, d_peaks_len, MAX_SIGNALS_PER_SPREAD, threshold, SPREADING_LENGTH * 2, SPREADING_BINS);
	cudaThreadSynchronize();

	int h_peaks[MAX_SIGNALS_PER_SPREAD];
	unsigned int h_peaks_len;

	cudaMemcpy(h_peaks, d_peaks, MAX_SIGNALS_PER_SPREAD * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_peaks_len, d_peaks_len, sizeof(unsigned int), cudaMemcpyDeviceToHost);


	cout << "found " << h_peaks_len << " peaks! ::" << endl;

	// at this point magnitudes have been detected that aren't in the padding
	// the padding is actually in the center of the SPREADING_LENGTH, but we want to visiualize the padding on the outside and the data on the inside

	if( h_peaks_len > 2 )
		int a = 4;

	vector<int> localMaximaPeaks;
	bool isLocalMaxima;

	// look at all peaks above the thresh and scan for local maxima
	for( unsigned int i = 0; i < std::min((unsigned)MAX_SIGNALS_PER_SPREAD, h_peaks_len); i++ )
	{
		cout << "index of peak " << i << " is " << h_peaks[i] << " with mag2 " << magnitude2(h_cts[h_peaks[i]]);

		isLocalMaxima = sampleIsLocalMaxima(h_cts, h_peaks[i], SPREADING_LENGTH * 2, SPREADING_BINS);

		// save the index
		if( isLocalMaxima )
		{
			localMaximaPeaks.push_back(h_peaks[i]);
			cout << " LOCAL MAX";
		}

		cout << endl;
	}






//
//	// peak detection
//	checkCudaErrors(cudaMemset(d_peak, 0, sizeof(float)));
//	cudaThreadSynchronize();
//	gpu_peak_detection(d_cts, d_peak, SPREADING_LENGTH * 2, SPREADING_BINS);
//	cudaThreadSynchronize();
//	cudaMemcpy(h_peak, d_peak, sizeof(float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
//
//	// cast back to float from "sortable integer"
//	unsigned a, b, c;
//	//a = *((unsigned*)h_peak);
//	//b = ((a >> 31) - 1) | 0x80000000;
//	//c = a ^ b;
//	float d;
//	//d = *((float*)&c);
//	(unsigned&)d = IFloatFlip((unsigned&)h_peak);
//
//	d = sqrt(d);
//
//	cout << "old style peak is " << d << endl;
//
//
//	float h_thrust_peak;
//	int h_thrust_peak_index;
//
//#ifdef DEBUG_POPDECONVOLVE_TIME
//	ptime t1, t2;
//	time_duration td, tLast;
//	t1 = microsec_clock::local_time();
//#endif
//
//
//	thrust_peak_detection(d_cts, d_mag_vec, &h_thrust_peak, &h_thrust_peak_index, SPREADING_LENGTH * 2, SPREADING_BINS);
//
//#ifdef DEBUG_POPDECONVOLVE_TIME
//	t2 = microsec_clock::local_time();
//	td = t2 - t1;
//	cout << "thrust did peak and index detection in " << td.total_microseconds() << "us." << endl;
//#endif
//
//
//
//
//	float h_thrust_d;
//
//	h_thrust_d = sqrt(h_thrust_peak);
//
//	if( h_thrust_d > 3.7e4 )
//		cout << "THRUST style peak is " << h_thrust_d << endl;


}

#ifdef UNIT_TEST

BOOST_AUTO_TEST_CASE( blahtest )
{
	complex<float>* cfc;
	ptime t1, t2;
	time_duration td, tLast;

	cfc = (complex<float>*)malloc(512*2*sizeof(complex<float>)); ///< pad

	t1 = microsec_clock::local_time();
	pop::PopProtADeconvolve::gpu_gen_pn_match_filter_coef( pop::m4k_001, cfc, 512, 512, 0.5 );
	t2 = microsec_clock::local_time();

	BOOST_CHECK( cfc );

	td = t2 - t1;

	cout << "gen_pn_match_filter_coef() time = " << td.total_microseconds() << "us" << endl;

	free(cfc);
}



#define RAND_BETWEEN(Min,Max)  (((double(rand()) / double(RAND_MAX)) * (Max - Min)) + Min)


class PopTestRandComplexSource : public PopSource<complex<float> >
{
public:
	PopTestRandComplexSource() : PopSource<complex<float> >("PopTestRandComplexSource") { }


	    void send_both(size_t count, size_t stamps, double start_time = -1, double time_inc_divisor = -1)
	    {

	    	complex<float> *b = get_buffer(count);
	    	PopTimestamp t[stamps];

	    	float min, max;
	    	max = 225000;
	    	min = -1 * max;


	    	// build msgs
	    	for( size_t i = 0; i < count; i++ )
	    	{
	    		b[i].real( RAND_BETWEEN(min, max) );
	    		b[i].imag( RAND_BETWEEN(min, max) );

//	    		cout << "(" << b[i].real() << "," << b[i].imag() << ")" << endl;
	    	}



	    	process(count);

	    }


};


BOOST_AUTO_TEST_CASE( cpu_peek_compare )
{
//	PopProtADeconvolve* deconvolve = new PopProtADeconvolve();
//	deconvolve->start_thread();
//
//	PopTestRandComplexSource source;
//
//	source.connect(*deconvolve);
//
//	// always seed with this value for repeatable results
//	srand(1380748793);
//
//	source.send_both(SPREADING_LENGTH,0);
//
//	// sleep for N second(s)
//	for( ;; )
//	{
//		boost::posix_time::microseconds workTime(1000000);
//		boost::this_thread::sleep(workTime);
//	}
}







#endif // UNIT_TEST

} // namespace pop

