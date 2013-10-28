/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_DECONVOLVE_CU__
#define __POP_DECONVOLVE_CU__

//#include <popComplex.h>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include "dsp/utils.hpp"

#include <cufft.h>
#include <cstdlib>
#include <time.h>
//#include "cuPrintf.cu"
//#include "shrUtils.h"
//#include "cutil_inline.h"
#include <dsp/common/poptypes.cuh>

#include <dsp/prota/popchanfilter.cuh>

using namespace std;


__global__ void threshold_detection(popComplex *in, int *out, unsigned int *outLen, int outLenMax, double thresholdSquared, int len, int fbins)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

//	int I = blockDim.x * gridDim.x;
//
//	int F = (I / len);
//	int f = (i / len) - (F / 2); // frequency
	int b = i % len; // fft bin

	double mag; // magnitude of peak

	// don't look for peaks in padding
	if( (b > (len / 4)) && (b <= (3 * len /4)) ) return;

	// take the magnitude of the detection
	mag = magnitude2(in[i]);

	// if the magnitude is below the thresh, return
	if( mag < thresholdSquared ) return;

	// atomicInc incriments the variable at the pointer only if the second param is larger than the stored variable
	// this variable always starts at 0 (set before the kernel launch)
	// the "old" value at the pointer location is returned, which is this thread's unique index into the output buffer
	int ourUniqueIndex = atomicInc(outLen, INT_MAX);

	// out of bounds, this is an OVERFLOW ie we found too many peaks
	// in this case we discard the data, but outLen is still incrimented.  we can check for overflow after the kernel launch is done
	if( ourUniqueIndex > outLenMax ) return;

	// save the index of our detection to the array
	out[ourUniqueIndex] = i;
}


extern "C"
{	


	void gpu_threshold_detection(popComplex* d_in, int* d_out, unsigned int *d_outLen, int outLenMax, double threshold, int len, int fbins, cudaStream_t* stream)
	{
		// reset this index of the largest detected peak to 0
		checkCudaErrors(cudaMemsetAsync(d_outLen, 0, sizeof(int), *stream));

		threshold_detection<<<fbins * 16, len / 16, 0, *stream>>>(d_in, d_out, d_outLen, outLenMax, (threshold*threshold), len, fbins);
//
//		checkCudaErrors(cudaThreadSynchronize());
//
//		cudaDeviceSynchronize();
	}




//	void gpu_peak_detection(popComplex* in, double* peak, int len, int fbins)
//	{
//		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
//		peak_detection<<<fbins * 16, len / 16>>>(in, peak, len);
//		cudaThreadSynchronize();
//	}


}

#endif
