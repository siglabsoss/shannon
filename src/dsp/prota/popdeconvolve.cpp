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


#include "dsp/utils.hpp"

#include <core/popexception.hpp>

#include <dsp/prota/popchanfilter.cuh>
#include <dsp/prota/popdeconvolve.cuh>
#include "dsp/prota/popdeconvolve.hpp"
#include "core/basestationfreq.h"

//#define DEBUG_POPDECONVOLVE_TIME

using namespace std;
using namespace boost::posix_time;

namespace pop
{

PopTimestamp last;




extern "C" void gpu_rolling_dot_product(popComplex *in, popComplex *cfc, popComplex *out, int len, int fbins, cudaStream_t* stream );
extern "C" void gpu_peak_detection(popComplex* in, double* peak, int len, int fbins);
extern "C" void thrust_peak_detection(popComplex* in, thrust::device_vector<double>* d_mag_vec, double* peak, int* index, int len, int fbins);
extern "C" void init_popdeconvolve(thrust::device_vector<double>** d_mag_vec, size_t size);


PopProtADeconvolve::PopProtADeconvolve() : PopSinkGpu<complex<double>[CHANNELS_USED] >( "PopProtADeconvolve", SPREADING_LENGTH ),
		cts( "PopProtADeconvolve" ), maxima ("PopProtADeconvolveMaxima"), peaks ("PopProtADeconvolvePeaks"),
		// this multiplier gives us the ability to hold 46080 samples which is enough to negatively index 542*80 (43360)
		// get this number by looking at the printed output and doing this equation ( (usable KiB * 1024) / 2 ) / datatype
		cts_mag_gpu ("CtsForAllChannelsAndCodes GPU", SPREADING_LENGTH, 180)
{

}

PopProtADeconvolve::~PopProtADeconvolve()
{
	cufftDestroy(plan_fft);
	cufftDestroy(many_plan_fft);
	cufftDestroy(plan_deconvolve);
	checkCudaErrors(cudaFree(d_sts));
	checkCudaErrors(cudaFree(d_sfs));
	checkCudaErrors(cudaFree(d_cfc));
	checkCudaErrors(cudaFree(d_cfs));
	checkCudaErrors(cudaFree(d_cts));
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

    // create CUDA stream
    checkCudaErrors(cudaStreamCreate(&deconvolve_stream));


    // setup FFT plans
    int dimension_size = SPREADING_LENGTH * 2;

    cufftPlanMany(&many_plan_fft, 1, &dimension_size,
    		&dimension_size, CHANNELS_USED, 1,
    		&dimension_size, 1, SPREADING_LENGTH * 2,
    		CUFFT_Z2Z, CHANNELS_USED); // pad

    cufftPlan1d(&plan_fft, SPREADING_LENGTH * 2, CUFFT_Z2Z, 1); // pad
    cufftPlanMany(&plan_deconvolve, 1, &dimension_size,
    		0, 1, 0,
    		0, 1, 0,
    		CUFFT_Z2Z, SPREADING_BINS);

    // assign plans to a stream
    cufftSafeCall(cufftSetStream(many_plan_fft,   deconvolve_stream));
    cufftSafeCall(cufftSetStream(plan_deconvolve, deconvolve_stream));

    // allocate device memory
    checkCudaErrors(cudaMalloc(&d_sts, CHANNELS_USED * SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_sfs, CHANNELS_USED * SPREADING_LENGTH * 2 * sizeof(popComplex)));
    // host has an array of pointers which will point to d_cfc's.  after this cuda malloc we aren't quit done yet
    checkCudaErrors(cudaMalloc(&d_cfc[0], SPREADING_LENGTH * 2 * SPREADING_CODES * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_cfs, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMalloc(&d_cts, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));



    // initialize device memory
    checkCudaErrors(cudaMemset(d_sts, 0, CHANNELS_USED * SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_sfs, 0, CHANNELS_USED * SPREADING_LENGTH * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cfc[0], 0, SPREADING_LENGTH * 2 * SPREADING_CODES * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cfs, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));
    checkCudaErrors(cudaMemset(d_cts, 0, SPREADING_LENGTH * SPREADING_BINS * 2 * sizeof(popComplex)));

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


double PopProtADeconvolve::sincInterpolateMaxima(popComplex* data, int sample, int neighbors )
{
	size_t osl; // osl oversampled symbol length
	osl = PEAK_SINC_SAMPLES; // should be one hundred thousand
	const complex<double>* y = (complex<double>*)data; // rename this symbol

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

	static double sina_a[PEAK_SINC_SAMPLES][PEAK_SINC_NEIGHBORS+PEAK_SINC_NEIGHBORS+1];
	static bool computed = false;

	// compute sin(a)/a matrix only first time
	// NOTE: This means this function cannot be called with neighbors != PEAK_SINC_NEIGHBORS
	if( !computed )
	{
		for( m = oslClipMin; m < oslClipMax; m++ )
		{
			for( n = 0; n < ncs; n++ )
			{
				a = M_PI * ( (double)m / (double)osl * (double)ncs - (double)n );
				if( 0 == a )
					sina_a[m][n] = 0.0;
				else
					sina_a[m][n] = sin(a) / a;
			}
		}

		computed = true;
	}



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
				yp[m] += sin(a) / a * y[n];
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




void PopProtADeconvolve::process(const std::complex<double> (*in)[CHANNELS_USED], size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size)
{

//	cout << "in deconv with " << len << " samples and " << timestamp_size << " stamps " << endl;

	// this is a running counter which represents the x-offset of the first valid sample (without padding) that will be unique for a long time
	static size_t running_counter = 0;

	unsigned n;
	double h_peak[10];

	ptime t1, t2;
	time_duration td, tLast;
	t1 = microsec_clock::local_time();

	//cout << "received " << len << " samples" << endl;

	if( len != SPREADING_LENGTH )
		throw PopException("size does not match filter");

	// copy new host data into device memory
//	cudaMemcpyAsync(d_sts, in - SPREADING_LENGTH, 50 * SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyHostToDevice, deconvolve_stream);
//	cudaThreadSynchronize();

	// perform FFTs on each channel (50)
	cufftExecZ2Z(many_plan_fft, (cufftDoubleComplex*)(in - SPREADING_LENGTH), (cufftDoubleComplex*)d_sfs, CUFFT_FORWARD);
//	cudaThreadSynchronize();

	checkCudaErrors(cudaGetLastError());


	double (*cts_mag_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS] = cts_mag_gpu.get_buffer();

	for( int channel = 0; channel < CHANNELS_USED; channel++ )
	{
		deconvolve_channel(channel, running_counter, cts_mag_buff, in, len, timestamp_data, timestamp_size);
	}

	running_counter += SPREADING_LENGTH;

	cts_mag_gpu.process();

	t2 = microsec_clock::local_time();
	td = t2 - t1;
	//cout << " PopDeconvolve - 1040 RF samples received and computed in " << td.total_microseconds() << "us." << endl;

}

PopTimestamp get_timestamp_for_index(double index, const PopTimestamp* timestamp_data)
{
	const PopTimestamp *prev = &timestamp_data[(int)floor(index)];
	const PopTimestamp *next = &timestamp_data[(int)floor(index)+1];

	// create mutable copy
	PopTimestamp timeDifference = PopTimestamp(*next);

	// calculate difference using -= overload (which should be most accurate)
	timeDifference -= *prev;

	double timePerSample = timeDifference.get_real_secs();

//	cout << "    with time per sample of " << boost::lexical_cast<string>(timePerSample) << endl;

	PopTimestamp exactTimestamp = PopTimestamp(*prev);

	//			//		cout << "    number of samples diff " << ( sincTimeIndex - prev->offset_adjusted(timestamp_buffer_correction) );
	//
	exactTimestamp += timePerSample * ( index - floor(index) );

	return exactTimestamp;
}


void PopProtADeconvolve::deconvolve_channel(unsigned channel, size_t running_counter, double (*cts_mag_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], const std::complex<double> (*in)[CHANNELS_USED], size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
//	complex<double>* h_cts = cts.get_buffer(len * SPREADING_BINS * 2);


	size_t channel_offset = SPREADING_LENGTH * 2 * channel;

	for(int spreading_code = 0; spreading_code < SPREADING_CODES; spreading_code++ )
	{

		// rolling dot product
		gpu_rolling_dot_product(d_sfs + channel_offset, d_cfc[spreading_code], d_cfs, SPREADING_LENGTH * 2, SPREADING_BINS, &deconvolve_stream);

		// perform IFFT on dot product for each bin.  data is layed out sequentially (SPREADING_LENGTH * 2 samples for bin 0, followed by bin 1... )
		cufftExecZ2Z(plan_deconvolve, (cufftDoubleComplex*)d_cfs, (cufftDoubleComplex*)d_cts, CUFFT_INVERSE);
		//cudaMemcpy(h_cts, d_cts, SPREADING_BINS * SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyDeviceToHost);

		checkCudaErrors(cudaGetLastError());


		cudaStreamSynchronize(deconvolve_stream); // move outside of deconvolve FIXME


		gpu_cts_stride_copy(cts_mag_buff, d_cts, channel, spreading_code, SPREADING_LENGTH * 2, SPREADING_BINS, &deconvolve_stream);


		cudaStreamSynchronize(deconvolve_stream); // move outside of deconvolve FIXME


//		double threshold = rbx::Config::get<double>("basestation_threshhold");
//
//
//		// allocate data in a rectangle which is +- neighbor samples and +- 1 fbin (which gives us *3)
//		popComplex h_cts[MAX_SIGNALS_PER_SPREAD * PEAK_SINC_SAMPLES_TOTAL * 3 ];
//		unsigned h_maxima_peaks[MAX_SIGNALS_PER_SPREAD];
//
//		unsigned h_maxima_peaks_len;
//
//		// threshold detection copies final data results to host
//		gpu_threshold_detection(d_cts, d_peaks, d_peaks_len, d_maxima_peaks, d_maxima_peaks_len, PEAK_SINC_NEIGHBORS, MAX_SIGNALS_PER_SPREAD, h_cts, h_maxima_peaks, &h_maxima_peaks_len, threshold, SPREADING_LENGTH * 2, SPREADING_BINS, &deconvolve_stream);
//
////		unsigned int h_peaks_len;
////		cudaMemcpyAsync(&h_peaks_len, d_peaks_len, sizeof(unsigned int), cudaMemcpyDeviceToHost, deconvolve_stream);
//
//		// at this point h_cts an array that contains d_cts samples in chunks of PEAK_SINC_SAMPLES_TOTAL (17) samples at a time.
//		// these chunks surround peaks.  The array isn't sparse, but it represents sparse data
//
//
//		// Grab a buffer from our second source in prepration for outputting local maxima peaks
//
//		PopPeak* peaksOut;
//		PopPeak* currentPeak;
//
//		// only get_buffer if we are going to write into it
//		if( h_maxima_peaks_len != 0 )
//		{
//			peaksOut = peaks.get_buffer( h_maxima_peaks_len );
//			cout << "got " << h_maxima_peaks_len << " peaks!" << endl;
//		}
//
//
//
//		int up = 0;
//		int center = PEAK_SINC_SAMPLES_TOTAL;
//		int down   = PEAK_SINC_SAMPLES_TOTAL*2;
//
//		for( unsigned int i = 0; i < std::min((unsigned)MAX_SIGNALS_PER_SPREAD, h_maxima_peaks_len); i++ )
//		{
//			// pointer to current maxima in the source buffer
//			currentPeak = peaksOut+i;
//
//
//			// calculate the center sample (peak) in units of the sparse h_cts array for this peak
//			// remember h_cts is a sparse array with 3 runs of PEAK_SINC_SAMPLES_TOTAL samples on the previous, detected, and next fbins
//			unsigned h_cts_peak_index = PEAK_SINC_NEIGHBORS + (PEAK_SINC_SAMPLES_TOTAL*3) * i + center;
//
//			double timeIndex;
//			int timeBin;
//
//			// loop through detected fbins (ignore previous and next)
//			for( unsigned int sample = PEAK_SINC_SAMPLES_TOTAL; sample < PEAK_SINC_SAMPLES_TOTAL*2; sample++ )
//			{
//				// h_cts is sparse, this index is the original index into the d_cts array
//				int d_cts_index = h_maxima_peaks[i] - PEAK_SINC_NEIGHBORS + (sample % PEAK_SINC_SAMPLES_TOTAL);
//
//				// use integer division math to bump us to the correct fbin
//				int fbin_bump = (sample / PEAK_SINC_SAMPLES_TOTAL) * SPREADING_LENGTH * 2;
//
//				d_cts_index += fbin_bump;
//
//
//				boost::tie(timeIndex, timeBin) = linearToBins(d_cts_index, SPREADING_LENGTH * 2, SPREADING_BINS);
//
////				cout << "i = " << i << " sample = " << sample << " timeIndex = " << timeIndex << " timeBin = " << timeBin << endl;
//
//				// set the data point and timestamp for this specific sample
//				// but offset so the PopPeak array only contains data for the detected fbin
//				currentPeak->data[sample - PEAK_SINC_SAMPLES_TOTAL].sample = h_cts[sample];
//				// FIXME: timestamps are still on the GPU
////				currentPeak->data[sample - PEAK_SINC_SAMPLES_TOTAL].timestamp = get_timestamp_for_index(timeIndex, timestamp_data);
//
//				// all the positional data in the PopPeak object is related to the upper left sample
//				// if we are on the first iteration of the loop, set this stuff now
//				if( sample == 0 )
//				{
//					currentPeak->sample_x = running_counter + timeIndex;
//					currentPeak->fbin = timeBin;
//				}
//			}
//
//			// finish this PopPeak
//			currentPeak->channel = channel;
//			currentPeak->symbol = spreading_code;
//			currentPeak->basestation = rbx::Config::get<double>("basestation_id");
//
//
//
//
////			cout << "code " << spreading_code << " peak number " << i << " found on channel " << channel << " in bin " << h_maxima_peaks[i] << " with mag " << sqrt(magnitude2(h_cts[h_cts_peak_index])) << endl;
//
//		}

		// call process and send out all detected peaks.  If this is 0 nothing happens (ie it's handled correctly internally)
//		peaks.process(h_maxima_peaks_len);
	}

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

