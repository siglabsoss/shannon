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


__global__ void threshold_detection(const double (*cts_stream)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], uint8_t(*out)[BYTES_PER_DETECTED_PACKET], unsigned int *outLen, int outLenMax, double thresholdSquared)
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

	double mag0a, mag1a, mag0b, mag1b; // magnitude^2 of peak


	// take the magnitude of the detection (mag0 is for either code 0 or 1 here)
	mag0a = cts_stream[sampletime][channel][code][bin];

	// if the magnitude is below the thresh, return
	if( mag0a < thresholdSquared ) return;

	int sampleOffset;
	// check previous N points to see if they are also above thresh
	for( int previous = 1; previous < 50; previous++ )
	{
		sampleOffset = sampletime - (previous * EXPECTED_PEAK_SEPARATION);

		// assuming that SPREADING_CODES is 2, look for peaks at previous times
		mag0a = cts_stream[sampleOffset][channel][0][bin];
		mag1a = cts_stream[sampleOffset][channel][1][bin];

		// compensate for drift by also checking in 1 previous
		sampleOffset -= 1;

		mag0b = cts_stream[sampleOffset][channel][0][bin];
		mag1b = cts_stream[sampleOffset][channel][1][bin];

		// if no time bin has a peak, bail
		if( mag0a < thresholdSquared && mag1a < thresholdSquared && mag0b < thresholdSquared && mag1b < thresholdSquared)
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
//	out[ourUniqueIndex] = i;

	uint8_t *storage = out[ourUniqueIndex];

//	uint8_t storage[BYTES_PER_DETECTED_PACKET];

//	uint8_t storage2;


	unsigned detectedBit = 0;

	// start from 0 this time
	for( int previous = 0; previous < 80; previous++ )
	{
		sampleOffset = sampletime - (previous * EXPECTED_PEAK_SEPARATION);

		// assuming that SPREADING_CODES is 2, look for peaks at previous times
		mag0a = cts_stream[sampleOffset][channel][0][bin];
		mag1a = cts_stream[sampleOffset][channel][1][bin];

		// compensate for drift by also checking in 1 previous
		sampleOffset -= 1;

		mag0b = cts_stream[sampleOffset][channel][0][bin];
		mag1b = cts_stream[sampleOffset][channel][1][bin];

		if( max(mag1b,mag1a) > max(mag0b,mag0a) )
			detectedBit = 1;

		pak_change_bit(storage, 79-previous, detectedBit);
	}

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
	void gpu_threshold_detection(const double (*cts_mag_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], uint8_t(*d_out)[BYTES_PER_DETECTED_PACKET], unsigned int *d_outLen, int outLenMax, uint8_t(*h_maxima_peaks)[BYTES_PER_DETECTED_PACKET], unsigned *h_maxima_peaks_len, double threshold, size_t sample_size, cudaStream_t* stream)
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

		threshold_detection<<<iterations/threadCount, threadCount, 0, *stream>>>(cts_mag_buff, (uint8_t(*)[BYTES_PER_DETECTED_PACKET])d_out, d_outLen, outLenMax, (threshold*threshold));

		checkCudaErrors(cudaGetLastError());

//		checkCudaErrors(cudaMemsetAsync(d_maxima_outLen, 0, sizeof(unsigned int), *stream));
//
//		local_maxima_detection<<<1, outLenMax, 0, *stream>>>(d_in, d_out, d_outLen, d_maxima_out, d_maxima_outLen, peak_sinc_neighbors, outLenMax, len, fbins);
//
//		checkCudaErrors(cudaGetLastError());
//
//		// copy the results back to the host
		cudaMemcpyAsync(h_maxima_peaks_len, d_outLen, sizeof(unsigned int), cudaMemcpyDeviceToHost, *stream);
		cudaMemcpyAsync(h_maxima_peaks, d_out, MAX_SIGNALS_PER_SPREAD * BYTES_PER_DETECTED_PACKET * sizeof(uint8_t), cudaMemcpyDeviceToHost, *stream);
//
//		// block till all actions on this stream have completed
		cudaStreamSynchronize(*stream);
//

//
//		// loop through results and copy neighboring samples back to host
		for(unsigned i = 0; i < *h_maxima_peaks_len; i++)
		{
			pak_print(h_maxima_peaks[i], 80);
			cout << endl << endl;
//			// copy left and right neighbors on the previous fbin
//			cudaMemcpyAsync(h_cts + up + i*(totalSamples*3),     d_in - len + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
//
//			// copy left and right neighbors on the fbin with detected local maxima peak
//			cudaMemcpyAsync(h_cts + center + i*(totalSamples*3), d_in + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
//
//			// copy left and right neighbors on the next fbin
//			cudaMemcpyAsync(h_cts + down + i*(totalSamples*3),   d_in + len + h_maxima_peaks[i] - peak_sinc_neighbors, sizeof(popComplex) * totalSamples, cudaMemcpyDeviceToHost, *stream);
		}
//
//		// block till all actions on this stream have completed
//		cudaStreamSynchronize(*stream);
	}

}

#endif
