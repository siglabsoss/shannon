/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BINNER_CU__
#define __POP_BINNER_CU__

#include <complex>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include "dsp/utils.hpp"

#include <cufft.h>
#include <cstdlib>
#include <time.h>

//#include <dsp/common/poptypes.cuh>

//#include <dsp/prota/popchanfilter.cuh>
#include <dsp/prota/popbinner.cuh>

using namespace std;


__global__ void threshold_detection(const popComplex (*cts_stream)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], int *out, unsigned int *outLen, int outLenMax, double thresholdSquared, int len, int fbins)
{
	int j;
	int k;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

//	int I = blockDim.x * gridDim.x;
//
//	int F = (I / len);
//	int f = (i / len) - (F / 2); // frequency
//	int b = i % len; // fft bin

	int bin = i % SPREADING_BINS;
	int code = (int)(i / SPREADING_BINS) % SPREADING_CODES;
	int channel = (int)(i / (SPREADING_BINS*SPREADING_CODES)) % CHANNELS_USED;
	int sampletime = (int)(i / (SPREADING_BINS*SPREADING_CODES * CHANNELS_USED));

	double mag0; // magnitude of peak
	double mag1; // magnitude of peak

	if( channel == 9 )
	{
		j = 666;
		k = j++;
	}

	// don't look for peaks in padding
//	if( (b > (len / 4)) && (b <= (3 * len /4)) ) return;

	// take the magnitude of the detection (mag0 is for either code 0 or 1 here)
	mag0 = magnitude2(cts_stream[sampletime][channel][code][bin]);

	// if the magnitude is below the thresh, return
	if( mag0 < thresholdSquared ) return;

	int sampleOffset;
	// check previous N points to see if they are also above thresh
	for( int previous = 1; previous < 10; previous++ )
	{
		sampleOffset = sampletime - (previous * EXPECTED_PEAK_SEPARATION);

		// assuming that SPREADING_CODES is 2, look for peaks at previous times
		mag0 = magnitude2(cts_stream[sampleOffset][channel][0][bin]);
		mag1 = magnitude2(cts_stream[sampleOffset][channel][1][bin]);

		// if neither time has a peak, bail
		if( mag0 < thresholdSquared && mag1 < thresholdSquared)
			return;
	}

	// if we've made it to here it means that the previous N time spots also had a peak in either code 0 or code 1


	// atomicInc increments the variable at the pointer only if the second param is larger than the stored variable
	// this variable always starts at 0 (set before the kernel launch)
	// the "old" value at the pointer location is returned, which is this thread's unique index into the output buffer
	int ourUniqueIndex = atomicInc(outLen, INT_MAX);

	// out of bounds, this is an OVERFLOW ie we found too many peaks
	// in this case we discard the data, but outLen is still incremented.  we can check for overflow after the kernel launch is done
	if( ourUniqueIndex > outLenMax ) return;

	// save the index of our detection to the array
	out[ourUniqueIndex] = i;

	j = k;
}

#define CHECK_POINTS (8)

// data is raw complex double float samples
// in is an array of detected peaks
// inLen is the count of detected peaks
// out is an array of detected local maxima pkeas
// outLen is the number of detected local maxima peaks

__global__ void local_maxima_detection(popComplex *data, int *in, unsigned int *inLen, int *out, unsigned int *outLen, unsigned peak_sinc_neighbors, int outLenMax, int spreadLength, int fbins)
{
	// detectedPeakIndex is the index into in[] which this thread is looking at
	int detectedPeakIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// don't process if this thread is looking at a peak that wasn't detected
	if( detectedPeakIndex >= *inLen )
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
}

#undef CHECK_POINTS



extern "C"
{	

// d_out is an array of samples which are above the threshold with size outLenMax
// d_outLen is the length of valid samples in the d_out array (with a value of no more than outLenMax)
// d_maxima_out is an array of samples which have passed the local maxima test
	void gpu_threshold_detection(const popComplex (*cts_stream_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, unsigned peak_sinc_neighbors, int outLenMax, popComplex* h_cts, unsigned *h_maxima_peaks, unsigned *h_maxima_peaks_len, double threshold, int len, int fbins, size_t sample_size, cudaStream_t* stream)
	{
		// reset this index of the largest detected peak to 0
		checkCudaErrors(cudaMemsetAsync(d_outLen, 0, sizeof(unsigned int), *stream));

		// how many actual popComplex samples did we just get
		// (sample_size is the number of [CHANNELS_USED][SPREADING_CODES][SPREADING_BINS] arrays we got)
		size_t iterations = sample_size * CHANNELS_USED * SPREADING_CODES * SPREADING_BINS;

//		cout << "iterations: " << iterations << endl;

		int threadCount = 512;

		if( iterations % threadCount != 0 )
			cout << "Error in math to calculate thread / block sizes." << endl;

		threshold_detection<<<iterations/threadCount, threadCount, 0, *stream>>>(cts_stream_buff, d_out, d_outLen, outLenMax, (threshold*threshold), len, fbins);

//		checkCudaErrors(cudaMemsetAsync(d_maxima_outLen, 0, sizeof(unsigned int), *stream));
//
//		local_maxima_detection<<<1, outLenMax, 0, *stream>>>(d_in, d_out, d_outLen, d_maxima_out, d_maxima_outLen, peak_sinc_neighbors, outLenMax, len, fbins);
//
//		checkCudaErrors(cudaGetLastError());
//
//		// copy the results back to the host
		cudaMemcpyAsync(h_maxima_peaks_len, d_outLen, sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
		cudaMemcpyAsync(h_maxima_peaks, d_maxima_out, sizeof(unsigned int) * outLenMax, cudaMemcpyDeviceToHost, *stream);
//
//		// block till all actions on this stream have completed
		cudaStreamSynchronize(*stream);
//
//		int totalSamples = (1+peak_sinc_neighbors+peak_sinc_neighbors);
//		int up = 0;
//		int center = totalSamples;
//		int down = totalSamples*2;
//
//
//		// loop through results and copy neighboring samples back to host
//		for(unsigned i = 0; i < *h_maxima_peaks_len; i++)
//		{
//			// copy left and right neighbors on the previous fbin
//			cudaMemcpyAsync(h_cts + up + i*(totalSamples*3),     d_in - len + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
//
//			// copy left and right neighbors on the fbin with detected local maxima peak
//			cudaMemcpyAsync(h_cts + center + i*(totalSamples*3), d_in + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
//
//			// copy left and right neighbors on the next fbin
//			cudaMemcpyAsync(h_cts + down + i*(totalSamples*3),   d_in + len + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
//		}
//
//		// block till all actions on this stream have completed
//		cudaStreamSynchronize(*stream);
	}

}

#endif
