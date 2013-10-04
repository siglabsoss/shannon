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

//#include <cuComplex.h>
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

#include <dsp/prota/popchanfilter.cuh>

using namespace std;


//__global__ void gpu_threshold_detection(cuComplex* in, int* out, int* outLen, int outMaxLen, int len, int fbins);

__global__ void threshold_detection(cuComplex *in, int *out, unsigned int *outLen, int outLenMax, float thresholdSquared, int len, int fbins)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

//	int I = blockDim.x * gridDim.x;
//
//	int F = (I / len);
//	int f = (i / len) - (F / 2); // frequency
	int b = i % len; // fft bin
	float mag; // magnitude of peak
	unsigned si; // sortable integer

	// don't look for peaks in padding
	if( (b > (len / 4)) && (b <= (3 * len /4)) ) return;

	// take the magnitude of the detection
	mag = magnitude2(in[i]);

	if( mag < thresholdSquared ) return;

	int oldVal = atomicInc(outLen, INT_MAX);

	if( oldVal > outLenMax ) return; // out of bounds, this is an OVERFLOW ie we found too many peaks

	out[oldVal] = i;

//	printf("here from cuda\r\n");

//	// transform into sortable integer
//	// https://devtalk.nvidia.com/default/topic/406770/cuda-programming-and-performance/atomicmax-for-float/
//	//si = *((unsigned*)&mag) ^ (-signed(*((unsigned*)&mag)>>31) | 0x80000000);
//	si = FloatFlip((unsigned&)mag);
//
//	// check to see if this is the highest recorded value
//	atomicMax((unsigned*)peak, si);
}


extern "C"
{	


	void gpu_threshold_detection(cuComplex* d_in, int* d_out, unsigned int *d_outLen, int outLenMax, float threshold, int len, int fbins)
	{
		cout << "here" << endl;

		// reset this counter to 0
		checkCudaErrors(cudaMemset(d_outLen, 0, sizeof(int)));

		threshold_detection<<<fbins * 16, len / 16>>>(d_in, d_out, d_outLen, outLenMax, (threshold*threshold), len, fbins);
//
//		checkCudaErrors(cudaThreadSynchronize());
//
//		cudaDeviceSynchronize();
	}




//	void gpu_peak_detection(cuComplex* in, float* peak, int len, int fbins)
//	{
//		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
//		peak_detection<<<fbins * 16, len / 16>>>(in, peak, len);
//		cudaThreadSynchronize();
//	}


}

#endif
