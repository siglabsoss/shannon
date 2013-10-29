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
#include <dsp/prota/popdeconvolve.cuh>

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

	// atomicInc increments the variable at the pointer only if the second param is larger than the stored variable
	// this variable always starts at 0 (set before the kernel launch)
	// the "old" value at the pointer location is returned, which is this thread's unique index into the output buffer
	int ourUniqueIndex = atomicInc(outLen, INT_MAX);

	// out of bounds, this is an OVERFLOW ie we found too many peaks
	// in this case we discard the data, but outLen is still incremented.  we can check for overflow after the kernel launch is done
	if( ourUniqueIndex > outLenMax ) return;

	// save the index of our detection to the array
	out[ourUniqueIndex] = i;
}

#define CHECK_POINTS (8)

// data is raw complex double float samples
// in is an array of detected peaks
// inLen is the count of detected peaks
// out is an array of detected local maxima pkeas
// outLen is the number of detected local maxima peaks

__global__ void local_maxima_detection(popComplex *data, int *in, unsigned int *inLen, int *out, unsigned int *outLen, popComplex* d_maxima_peaks_neighbors, unsigned peak_sinc_neighbors, int outLenMax, int spreadLength, int fbins)
{
	// detectedPeakIndex is the index into in[] which this thread is looking at
	int detectedPeakIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// don't process if this thread is looking at a peak that wasn't detected
	if( detectedPeakIndex > *inLen )
		return;

	// sample is the index into data which we are considering for a local maxima
	int sampleIndex = in[detectedPeakIndex];

	double surroundingMax = 0.0; // maximum of surrounding points

	int check[CHECK_POINTS];

	check[0] = sampleIndex + 1; // right
	check[1] = sampleIndex - 1; // left
	check[2] = sampleIndex - spreadLength; // up
	check[3] = sampleIndex + spreadLength; // down
	check[4] = sampleIndex - spreadLength - 1; // up left
	check[5] = sampleIndex - spreadLength + 1; // up right
	check[6] = sampleIndex + spreadLength - 1; // down left
	check[7] = sampleIndex + spreadLength + 1; // down right

	int testSample;

	for( int i = 0; i < CHECK_POINTS; i++ )
	{
		// bound to edges of array
		testSample = min(max(check[i],0), spreadLength*fbins);

		// compute maximum of surrounding indices
		surroundingMax = fmax(surroundingMax, magnitude2(data[testSample]));
	}


	// bail if not a local maxima
	if ( magnitude2(data[sampleIndex]) <= surroundingMax )
		return;


	// atomicInc increments the variable at the pointer only if the second param is larger than the stored variable
	// this variable always starts at 0 (set before the kernel launch)
	// the "old" value at the pointer location is returned, which is this thread's unique index into the output buffer
	int ourUniqueIndex = atomicInc(outLen, INT_MAX);

	// out of bounds, this is an OVERFLOW ie we found too many peaks
	// in this case we discard the data, but outLen is still incremented.  we can check for overflow after the kernel launch is done
	if( ourUniqueIndex > outLenMax ) return;

	// save the index of our detection to the array
	out[ourUniqueIndex] = sampleIndex;

	// copy neighbors into this array;
	// we copy peak_sinc_neighbors (8) samples on either side for a total of (17) samples
	// after this kernel is done the host can do a single memcopy and get all the needed information for sinc interpolation
	memcpy(d_maxima_peaks_neighbors, data+sampleIndex-peak_sinc_neighbors, (1+peak_sinc_neighbors+peak_sinc_neighbors)*sizeof(popComplex) );
}

#undef CHECK_POINTS


extern "C"
{	

// d_out is an array of samples which are above the threshold with size outLenMax
// d_outLen is the length of valid samples in the d_out array (with a value of no more than outLenMax)
// d_maxima_out is an array of samples which have passed the local maxima test
	void gpu_threshold_detection(popComplex* d_in, int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, popComplex* d_maxima_peaks_neighbors, unsigned peak_sinc_neighbors, int outLenMax, double threshold, int len, int fbins, cudaStream_t* stream)
	{
		// reset this index of the largest detected peak to 0
		checkCudaErrors(cudaMemsetAsync(d_outLen, 0, sizeof(int), *stream));

		threshold_detection<<<fbins * 16, len / 16, 0, *stream>>>(d_in, d_out, d_outLen, outLenMax, (threshold*threshold), len, fbins);

		checkCudaErrors(cudaMemsetAsync(d_maxima_outLen, 0, sizeof(int), *stream));

		local_maxima_detection<<<1, outLenMax, 0, *stream>>>(d_in, d_out, d_outLen, d_maxima_out, d_maxima_outLen, d_maxima_peaks_neighbors, peak_sinc_neighbors, outLenMax, len, fbins);

	}




//	void gpu_peak_detection(popComplex* in, double* peak, int len, int fbins)
//	{
//		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
//		peak_detection<<<fbins * 16, len / 16>>>(in, peak, len);
//		cudaThreadSynchronize();
//	}


}

#endif
