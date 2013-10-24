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
#include <algorithm>    // std::min
#include "boost/tuple/tuple.hpp"
#include <boost/lexical_cast.hpp>
#include "core/config.hpp"

#include "cuda/helper_cuda.h"

#include <core/popexception.hpp>

#include <dsp/prota/popchanfilter.cuh>
#include "popdeconvolve.hpp"

//#define DEBUG_POPDECONVOLVE_TIME

using namespace std;
using namespace boost::posix_time;

namespace pop
{

PopTimestamp last;


#define MAX_SIGNALS_PER_SPREAD (32) // how much memory to allocate for detecting signal peaks
#define PEAK_SINC_NEIGHBORS (8)     // how many samples to add to either side of a local maxima for sinc interpolate
#define PEAK_SINC_SAMPLES (100000)  // how many samples to sinc interpolate around detected peaks

extern "C" void gpu_rolling_dot_product(popComplex *in, popComplex *cfc, popComplex *out, int len, int fbins);
extern "C" void gpu_peak_detection(popComplex* in, double* peak, int len, int fbins);
extern "C" void gpu_threshold_detection(popComplex* d_in, int* d_out, unsigned int *d_outLen, int outLenMax, double threshold, int len, int fbins);
extern "C" void thrust_peak_detection(popComplex* in, thrust::device_vector<double>* d_mag_vec, double* peak, int* index, int len, int fbins);
extern "C" void init_popdeconvolve(thrust::device_vector<double>** d_mag_vec, size_t size);


PopProtADeconvolve::PopProtADeconvolve() : PopSink<complex<double> >( "PopProtADeconvolve", SPREADING_LENGTH ),
		cts( "PopProtADeconvolve" ), maxima ("PopProtADeconvolveMaxima")
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
	checkCudaErrors(cudaFree(d_peaks));
	checkCudaErrors(cudaFree(d_peaks_len));
}

	/// spreading code m4k_001
	const int8_t m4k_001[] = {-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1, 1,-1, 1,-1,-1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1};
	const int8_t s4k_001[] = {1, -1,-1, 1, 1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,-1,1,1,1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,1,1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,1,1,1,1,1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,1,-1,1};

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
	const int8_t* prn, complex<double>* cfc,
	size_t  ncs, size_t osl, double bt)
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

complex<double>* h_cfc;

void PopProtADeconvolve::init()
{
	last = PopTimestamp(0,0);

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
    cufftPlan1d(&plan_fft, SPREADING_LENGTH * 2, CUFFT_Z2Z, 1); // pad
    int rank_size = SPREADING_LENGTH * 2;
    cufftPlanMany(&plan_deconvolve, 1, &rank_size, 0, 1, 0, 0, 1, 0, CUFFT_Z2Z, SPREADING_BINS);

    // allocate device memory
    checkCudaErrors(cudaMalloc(&d_sts, SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_sfs, SPREADING_LENGTH * 2 * sizeof(popComplex)));
    // host has an array of pointers which will point to d_cfc's.  after this cuda malloc we aren't quit done yet
    checkCudaErrors(cudaMalloc(&d_cfc[0], SPREADING_LENGTH * 2 * SPREADING_CODES * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_cfs, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_cts, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_peaks, MAX_SIGNALS_PER_SPREAD * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_peaks_len, sizeof(unsigned int)));


    // initialize device memory
    checkCudaErrors(cudaMemset(d_sts, 0, SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_sfs, 0, SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cfc[0], 0, SPREADING_LENGTH * 2 * SPREADING_CODES * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cfs, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cts, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_peaks, 0, MAX_SIGNALS_PER_SPREAD * sizeof(int)));

    // malloc host memory
    d_sinc_yp = (complex<double>*)malloc(PEAK_SINC_SAMPLES * sizeof(complex<double>));

    // generate convolution filter coefficients
    cout << "generating spreading codes..." << endl;
    h_cfc = (complex<double>*)malloc(2 * SPREADING_LENGTH * sizeof(complex<double>));



    // list of spreading codes
    const int8_t* code_list[SPREADING_CODES];
    code_list[0] = s4k_001;
    code_list[1] = m4k_001;


    for( int i = 0; i < SPREADING_CODES; i++)
    {
    	// finish up our cuda malloc for d_cfc
    	// this line sets a pointer to the chunk of memory that we allocated on the device for each entry
        d_cfc[i] = d_cfc[0] + (SPREADING_LENGTH * 2) * i;

    	cout << "code " << i << endl;

    	// call this function which computes (and overwrites) the result into h_cfc on the host
    	gpu_gen_pn_match_filter_coef(code_list[i], h_cfc, SPREADING_LENGTH, SPREADING_LENGTH /*oversampled*/, 0.5);

    	// copy to the specific index of d_cfc on the device
    	checkCudaErrors(cudaMemcpy(d_cfc[i], h_cfc, 2 * SPREADING_LENGTH * sizeof(popComplex), cudaMemcpyHostToDevice));
    }

	cout << "done generating spreading codes" << endl;
	free(h_cfc);
}


unsigned IFloatFlip(unsigned f)
{
	unsigned mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}



double magnitude2( const complex<double>& in )
{
	return in.real() * in.real() + in.imag() * in.imag();
}

// we need to check the points forwards and backwards in time, we also need to check up and down in fbins
// then we also check the corners
bool sampleIsLocalMaxima(const complex<double>* data, int sample, int spreadLength, int fbins)
{
	//TODO: we don't have a circular buffer of the final data, so if a sample lies on a boundary
	// we can't check for local maxima by negativly indexing to verify local maximums

	int sampleTime, sampleBin;

	double surroundingMax = 0.0; // maximum of surrounding points

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


double PopProtADeconvolve::sincInterpolateMaxima(const complex<double>* data, int sample, int neighbors )
{
	size_t osl; // osl oversampled symbol length
	osl = PEAK_SINC_SAMPLES; // should be one hundred thousand
	const complex<double>* y = data; // rename this symbol

	// offset to first symbol
	y += sample - neighbors;

	size_t ncs; // number of input samples
	ncs = neighbors*2 + 1;
	size_t n, m;
	double a; // holder

	size_t pixelSize = ceil((double)(osl / ncs) * 1.05);  // number of oversamples that signify one pixel (padded by a margin)


	// we only sink interpolate and look for maximum between these values
	// this is the center plus and minus one pixel
	size_t oslClipMin, oslClipMax;
	oslClipMin = floor((double)(osl/2) - pixelSize);
	oslClipMax = ceil((double)(osl/2) + pixelSize);


	// rename member variable that was mallocd during init
	// Note that because we are only looking in the clipped region but using real offsets into the output array, we are mallocing a bunch of extra memory that is never used
	complex<double> *yp = d_sinc_yp;

	double img, rl;

	for( m = oslClipMin; m < oslClipMax; m++ )
	{
		yp[m] = complex<double>(0.0, 0.0);
		for( n = 0; n < ncs; n++ )
		{
			a = M_PI * ( (double)m / (double)osl * (double)ncs - (double)n );
			if( 0 == a )
				yp[m] += y[n];
			else
			{
				// required to keep things in double
				complex<double> holder(y[n].real(), y[n].imag());
				yp[m] += sin(a) / a * holder;
			}
		}

//		cout << "yp[" << m << "] = " << yp[m] << "( " << magnitude2(yp[m]) << " )" << endl;

//		cout << magnitude2(yp[m]) << endl;
	}

//	cout << endl << endl << endl;



	// max detection
	double max = 0.0;
	int maxIndex = 0;
	for( m = oslClipMin; m < oslClipMax; m++ )
	{
		max = std::max(magnitude2(yp[m]), max);

		if( max == magnitude2(yp[m]) )
			maxIndex = m;
	}

//	cout << "found max " << max << " at index " << maxIndex << endl;

	double decimatedIndex = (double)maxIndex / osl * ncs;

	if( decimatedIndex < (neighbors-0.5) )
		cout << "sincInterpolateMaxima out of bounds min" << endl;

	if( decimatedIndex > (neighbors+0.5) )
		cout << "sincInterpolateMaxima out of bounds max" << endl;


//	cout << "decimatedIndex " << decimatedIndex << endl;

	double sampleIndex = sample - ((ncs-1)/2.0) + decimatedIndex;

//	cout << "sampleIndex " << boost::lexical_cast<string>(sampleIndex) << endl;

	return sampleIndex;

}



// assumes input has not been shifted and is in the form of [m -> h][l -> m] where the center half is bad data
// assumes spreadLength to be 2x the number of valid samples
// returns a pair ( time , bin )
boost::tuple<double, int> linearToBins(double sample, int spreadLength, int fbins)
{
	double timeIndex;
	int fbin;

	// add half of spread length to offset us to the center, then mod
	// this leaves us with [l -> m][m -> h] where the first and last quarters are bad data
	timeIndex = fmod( sample + (spreadLength/2), spreadLength );

	// subtract 1 quarter to shift the good data into the first half.
	// this ranges the timeIndex from (0 - spreadLength/2)
	timeIndex -= (spreadLength / 4);

	fbin = floor(sample / spreadLength);

	return boost::tuple<double,int>(timeIndex,fbin);
}




void PopProtADeconvolve::process(const complex<double>* in, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
{

//	cout << "in deconv with " << len << " samples and " << timestamp_size << " stamps " << endl;

	unsigned n;
	double h_peak[10];

	ptime t1, t2;
	time_duration td, tLast;
	t1 = microsec_clock::local_time();

	//cout << "received " << len << " samples" << endl;

	if( len != SPREADING_LENGTH )
		throw PopException("size does not match filter");


//	for(size_t i = 0; i < timestamp_size; i++ )
//	{
////		timestampOut[i] = PopTimestamp(timestamp_data[i], calc_timestamp_offset(timestamp_data[i].offset, timestamp_buffer_correction) * factor );
//
//		cout << "got timestamp with raw index " << timestamp_data[i].offset << " and adjustment " << timestamp_buffer_correction << " which adjusts to " << calc_timestamp_offset(timestamp_data[i].offset, timestamp_buffer_correction) << endl;
////		cout << "got timestamp with raw index " << timestampOut[i].offset << " and ... " << endl;
//	}

//	cout << "  got " << timestamp_size << " timestamps for " << len << " samples." << endl;
//	cout << "  with indices " << timestamp_data[0].offset_adjusted(timestamp_buffer_correction) << " and " << timestamp_data[timestamp_size-1].offset_adjusted(timestamp_buffer_correction) << endl;


	complex<double>* h_cts = cts.get_buffer(len * SPREADING_BINS * 2);

	// copy new host data into device memory
	cudaMemcpy(d_sts, in - SPREADING_LENGTH, SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	// perform FFT on spectrum
	cufftExecZ2Z(plan_fft, (cufftDoubleComplex*)d_sts, (cufftDoubleComplex*)d_sfs, CUFFT_FORWARD);
	cudaThreadSynchronize();


	for(int spreading_code = 0; spreading_code < SPREADING_CODES; spreading_code++ )
	{

		// rolling dot product
		gpu_rolling_dot_product(d_sfs, d_cfc[spreading_code], d_cfs, SPREADING_LENGTH * 2, SPREADING_BINS);
		cudaThreadSynchronize();


		// perform IFFT on dot product
		cufftExecZ2Z(plan_deconvolve, (cufftDoubleComplex*)d_cfs, (cufftDoubleComplex*)d_cts, CUFFT_INVERSE);
		cudaThreadSynchronize();
		cudaMemcpy(h_cts, d_cts, SPREADING_BINS * SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();

		double threshold = rbx::Config::get<double>("basestation_threshhold");

		// threshold detection
		gpu_threshold_detection(d_cts, d_peaks, d_peaks_len, MAX_SIGNALS_PER_SPREAD, threshold, SPREADING_LENGTH * 2, SPREADING_BINS);
		cudaThreadSynchronize();

		int h_peaks[MAX_SIGNALS_PER_SPREAD];
		unsigned int h_peaks_len;

		cudaMemcpy(h_peaks, d_peaks, MAX_SIGNALS_PER_SPREAD * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_peaks_len, d_peaks_len, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		//	cout << "found " << h_peaks_len << " peaks! ::" << endl;
		// at this point magnitudes have been detected that aren't in the padding

		vector<int> localMaximaPeaks;
		bool isLocalMaxima;

		// look at all peaks above the thresh and scan for local maxima
		for( unsigned int i = 0; i < std::min((unsigned)MAX_SIGNALS_PER_SPREAD, h_peaks_len); i++ )
		{
			//		cout << "index of peak " << i << " is " << h_peaks[i] << " with mag2 " << magnitude2(h_cts[h_peaks[i]]);

			isLocalMaxima = sampleIsLocalMaxima(h_cts, h_peaks[i], SPREADING_LENGTH * 2, SPREADING_BINS);

			// save the index
			if( isLocalMaxima )
			{
				localMaximaPeaks.push_back(h_peaks[i]);
				//			cout << " LOCAL MAX";
			}

			//		cout << endl;
		}



		// Grab a buffer from our second source in prepration for outputting local maxima peaks

		PopSymbol* maximaOut;
		PopSymbol* currentMaxima;

		// only get_buffer if we are going to write into it
		if( localMaximaPeaks.size() != 0 )
			maximaOut = maxima.get_buffer( localMaximaPeaks.size() );

//		cout << "got " << localMaximaPeaks.size() << " peaks!" << endl;


		for( unsigned i = 0; i < localMaximaPeaks.size(); i++ )
		{
//			cout << endl;
//			cout << endl;
//			cout << endl;

			// calculate the fractional sample which the peak occured in relative to the linear sample
			double sincIndex = sincInterpolateMaxima(h_cts, localMaximaPeaks[i], PEAK_SINC_NEIGHBORS);
			//		cout << "sincIndex " << boost::lexical_cast<string>(sincIndex) << endl;

			double sincTimeIndex;
			int sincTimeBin;

			// convert this linear sample into a range of (0 - SPREADING_LENGTH) samples which represents real time
			boost::tie(sincTimeIndex, sincTimeBin) = linearToBins(sincIndex, SPREADING_LENGTH * 2, SPREADING_BINS);

//			cout << "sincTimeIndex " << sincTimeIndex << " ( " << 100 * sincTimeIndex / (SPREADING_LENGTH * 2) << "% )" << " sincTimeBin " << sincTimeBin << endl;

			const PopTimestamp *prev = &timestamp_data[(int)floor(sincTimeIndex)];
			const PopTimestamp *next = &timestamp_data[(int)floor(sincTimeIndex)+1];

//
//			// loop through every timestamp except for the last one and set the prev/next pointers to timestamps surrounding sincTimeIndex
//			for( size_t j = 0; j < timestamp_size - 1; j++ )
//			{
//				// grab the indices for j and j+1 (won't overflow because we never do the last iteration of the loop)
//				double currentIndex = timestamp_data[ j ].offset_adjusted(timestamp_buffer_correction);
//				double nextIndex    = timestamp_data[j+1].offset_adjusted(timestamp_buffer_correction);
//
//				//			cout << "    " << currentIndex;
//
//
//
//				// true when the loop is pointing at a timestamp with an largest index that is less than sincTimeIndex (assuming timestamps are in order)
//				if( nextIndex >= sincTimeIndex && prev == 0)
//				{
//					prev = timestamp_data + j; // set the pointer to this timestamp because the next is past
//					next = timestamp_data + j + 1; // set the pointer to the next timestamp because we just detected that it's past
//					break;
//				}
//
//				//			if( currentIndex > sincTimeIndex )
//				//			{
//				//				indexNext = std::min(indexNext, currentIndex);
//				//				cout << "n";
//				//			}
//
//				//			cout << endl;
//			}
//
//			//		cout << "    found min, max indexes of " << indexPrev << " // " << indexNext << endl;
//			//		cout << "    found min, max indexes of " << prev->offset_adjusted(timestamp_buffer_correction) << " /-/ " << next->offset_adjusted(timestamp_buffer_correction) << endl;

			// create mutable copy
			PopTimestamp timeDifference = PopTimestamp(*next);

			// calculate difference using -= overload (which should be most accurate)
			timeDifference -= *prev;

			double timePerSample = timeDifference.get_real_secs();// / (  next->offset_adjusted(timestamp_buffer_correction) - prev->offset_adjusted(timestamp_buffer_correction) );

			//		cout << "    with time per sample of " << boost::lexical_cast<string>(timePerSample) << endl;

			PopTimestamp exactTimestamp = PopTimestamp(*prev);



			//		cout << "    number of samples diff " << ( sincTimeIndex - prev->offset_adjusted(timestamp_buffer_correction) );

			exactTimestamp += timePerSample * ( sincTimeIndex - floor(sincTimeIndex) );

			cout << "code " << spreading_code << " peak number " << i << " found in bin " << sincTimeBin << " with mag " << sqrt(magnitude2(h_cts[localMaximaPeaks[i]])) << endl;

//					cout << "    prev time was" << boost::lexical_cast<string>(prev->get_real_secs()) << endl;
			cout << "    real time is " << boost::lexical_cast<string>(exactTimestamp.get_full_secs()) << "   -   " << boost::lexical_cast<string>(exactTimestamp.get_frac_secs()) << endl;
//			cout << boost::lexical_cast<string>(exactTimestamp.get_full_secs()) << ", " << boost::lexical_cast<string>(exactTimestamp.get_frac_secs()) << endl;


			if( i == 0 )
			{
				PopTimestamp delta = exactTimestamp;

				delta -= last;

				cout << "    delt time is " << boost::lexical_cast<string>(delta.get_full_secs()) << "   -   " << boost::lexical_cast<string>(delta.get_frac_secs()) << endl;


				last = exactTimestamp;
			}


			// pointer to current maxima in the source buffer
			currentMaxima = maximaOut+i;

			*currentMaxima = pop::PopSymbol(spreading_code, sqrt(magnitude2(h_cts[localMaximaPeaks[i]])), sincTimeBin, 0, rbx::Config::get<double>("basestation_id"), exactTimestamp);
		}

		// call process and send out all detected maxima.  If this is 0 nothing happens (ie it's handled correctly internally)
		maxima.process(localMaximaPeaks.size());

	}


	t2 = microsec_clock::local_time();
	td = t2 - t1;
	//cout << " PopDeconvolve - 1040 RF samples received and computed in " << td.total_microseconds() << "us." << endl;




//
//	// peak detection
//	checkCudaErrors(cudaMemset(d_peak, 0, sizeof(double)));
//	cudaThreadSynchronize();
//	gpu_peak_detection(d_cts, d_peak, SPREADING_LENGTH * 2, SPREADING_BINS);
//	cudaThreadSynchronize();
//	cudaMemcpy(h_peak, d_peak, sizeof(double), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
//
//	// cast back to double from "sortable integer"
//	unsigned a, b, c;
//	//a = *((unsigned*)h_peak);
//	//b = ((a >> 31) - 1) | 0x80000000;
//	//c = a ^ b;
//	double d;
//	//d = *((double*)&c);
//	(unsigned&)d = IFloatFlip((unsigned&)h_peak);
//
//	d = sqrt(d);
//
//	cout << "old style peak is " << d << endl;
//
//
//	double h_thrust_peak;
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
//	double h_thrust_d;
//
//	h_thrust_d = sqrt(h_thrust_peak);
//
//	if( h_thrust_d > 3.7e4 )
//		cout << "THRUST style peak is " << h_thrust_d << endl;


}

#ifdef UNIT_TEST

BOOST_AUTO_TEST_CASE( blahtest )
{
	complex<double>* cfc;
	ptime t1, t2;
	time_duration td, tLast;

	cfc = (complex<double>*)malloc(512*2*sizeof(complex<double>)); ///< pad

	t1 = microsec_clock::local_time();
	pop::PopProtADeconvolve::gpu_gen_pn_match_filter_coef( pop::m4k_001, cfc, 512, 512, 0.5 );
	t2 = microsec_clock::local_time();

	BOOST_CHECK( cfc );

	td = t2 - t1;

	cout << "gen_pn_match_filter_coef() time = " << td.total_microseconds() << "us" << endl;

	free(cfc);
}



#define RAND_BETWEEN(Min,Max)  (((double(rand()) / double(RAND_MAX)) * (Max - Min)) + Min)


class PopTestRandComplexSource : public PopSource<complex<double> >
{
public:
	PopTestRandComplexSource() : PopSource<complex<double> >("PopTestRandComplexSource") { }


	    void send_both(size_t count, size_t stamps, double start_time = -1, double time_inc_divisor = -1)
	    {

	    	complex<double> *b = get_buffer(count);
	    	PopTimestamp t[stamps];

	    	double min, max;
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

