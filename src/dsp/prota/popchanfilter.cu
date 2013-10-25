/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// Bad practice, but I can't get multiple cuda files to link http://stackoverflow.com/questions/13683575/cuda-5-0-separate-compilation-of-library-with-cmake
#include <dsp/prota/popdeconvolve.cu>
#include <dsp/common/poptypes.cu>


#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include "dsp/utils.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <time.h>

#include <dsp/common/poptypes.cuh>

#include <dsp/prota/popchanfilter.cuh>

#include "core/basestationfreq.h"



using namespace std;
using namespace thrust;


__global__ void rolling_scalar_multiply(popComplex *in, popComplex *cfc, popComplex *out, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int I = blockDim.x * gridDim.x;
	
	int fsearchbin = (I / len);
	int fidx = (i / len) - (fsearchbin / 2); // frequency modulation index
	int b = i % len; // fft bin
	int cidx = (b + fidx + len) % len;

	out[i] = in[b] * cfc[cidx];
}

__device__ unsigned IFloatFlip(unsigned f)
{
	unsigned mask = ((f >> 31) - 1) | 0x80000000;
	return f ^ mask;
}

__device__ unsigned FloatFlip(unsigned f)
{
	unsigned mask = -signed(f >> 31) | 0x80000000;
	return f ^ mask;
}



extern "C"
{	
	popComplex* d_dataa;
	popComplex* d_datab;
	popComplex* d_datac;
	popComplex* d_datad;
	cufftHandle plan1;
	cufftHandle plan2;
	size_t g_len_chan; ///< length of time series in samples
	size_t g_len_fft; ///< length of fft in samples
	size_t g_start_chan;


	size_t gpu_channel_split(const complex<double> *h_data, complex<double> *out)
	{
		//double ch_start, ch_end, ch_ctr;

		g_start_chan = bsf_channel_fbin_low(9);

/*		ch_start = 903626953 + (3200000 / (double)g_len_fft * (double)g_start_chan) - 1600000;
		ch_end = 903626953 + (3200000 / (double)g_len_fft * ((double)g_start_chan + 1040)) - 1600000;
		ch_ctr = (ch_start + ch_end) / 2.0;*/
		//printf("channel start: %f (%llu), end: %f, ctr: %f\r\n", ch_start, g_start_chan, ch_end, ch_ctr);

		// shift zero-frequency component to center of spectrum
		unsigned small_bin_start = (g_start_chan + (g_len_fft/2)) % g_len_fft;

		// calculate small bin side-band size
		unsigned small_bin_sideband = g_len_chan / 2;

		// copy new host data into device memory
		cudaMemcpy(d_dataa, h_data, g_len_fft * sizeof(popComplex), cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		// perform FFT on spectrum
		cufftExecZ2Z(plan1, (cufftDoubleComplex*)d_dataa, (cufftDoubleComplex*)d_datab, CUFFT_FORWARD);
		cudaThreadSynchronize();

		
		// chop spectrum up into 50 spreading channels low side-band
		cudaMemcpy(d_datac,
			       d_datab + small_bin_start + small_bin_sideband,
			       small_bin_sideband * sizeof(popComplex),
			       cudaMemcpyDeviceToDevice);
		// chop spectrum up into 50 spreading channels high side-band
		cudaMemcpy(d_datac + small_bin_sideband,
			       d_datab + small_bin_start,
			       small_bin_sideband * sizeof(popComplex),
			       cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();


		// put back into time domain
		cufftExecZ2Z(plan2, (cufftDoubleComplex*)d_datac, (cufftDoubleComplex*)d_datad, CUFFT_INVERSE);
		cudaThreadSynchronize();
  		checkCudaErrors(cudaGetLastError());

  		// Copy results to host
		cudaMemcpy(out, d_datad, g_len_chan * sizeof(popComplex), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		
  		return 0;
	}


	void init_deconvolve(size_t len_fft, size_t len_chan)
	{
		g_len_chan = len_chan;
		g_len_fft = len_fft;

		// allocate CUDA memory
		checkCudaErrors(cudaMalloc(&d_dataa, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMalloc(&d_datab, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMalloc(&d_datac, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMalloc(&d_datad, g_len_fft * sizeof(popComplex)));

		// initialize CUDA memory
		checkCudaErrors(cudaMemset(d_dataa, 0, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMemset(d_datab, 0, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMemset(d_datac, 0, g_len_fft * sizeof(popComplex)));
		checkCudaErrors(cudaMemset(d_datad, 0, g_len_fft * sizeof(popComplex)));

	    // setup FFT plans
	    cufftPlan1d(&plan1, g_len_fft, CUFFT_Z2Z, 1);
	    cufftPlan1d(&plan2, g_len_chan, CUFFT_Z2Z, 1);

	    printf("\n[Popwi::popprotadespread]: init deconvolve complete \n");
	}


	//Free all the memory that we allocated
	//TODO: check that this is comprehensive
	void cleanup() {
	  cufftDestroy(plan1);
	  cufftDestroy(plan2);
	  checkCudaErrors(cudaFree(d_dataa));
	  checkCudaErrors(cudaFree(d_datab));
	  checkCudaErrors(cudaFree(d_datac));
	  checkCudaErrors(cudaFree(d_datad));
	}


	void gpu_rolling_dot_product(popComplex *in, popComplex *cfc, popComplex *out, int len, int fbins)
	{
		// TODO: better refactor thread and block sizes for any possible spreading code and fbin lengths
		rolling_scalar_multiply<<<fbins * 16, len / 16>>>(in, cfc, out, len);
		cudaThreadSynchronize();
	}


	// this is the functor which calculates magnitude's for samples in the keep zone
	// and calculates 0.0 for samples outside of the zone
	// note for some weird reason if this struct has a normal style constructor other basic CUDA functions are affected and refuse to run!??
	struct indexed_magnitude_squared_functor_fixed : public thrust::binary_function<int,popComplex,double>
		{
		public:
			int m_len;

			__host__ __device__
			double operator()(const int& index, const popComplex& a) const {

				int b = index % m_len; // fft bin

				// if we in the region we want to cutoff, return 0.0 for the magnitude
				if( (b > (m_len / 4)) && (b <= (3 * m_len /4)) )
				{
					return 0.0;
				}

				return a.x * a.x + a.y * a.y;
			}
		};

	void thrust_peak_detection(popComplex* in, thrust::device_vector<double>* d_mag_vec, double* peak, int* index, int len, int fbins)
	{
		int totalLen = len*fbins;

		// grab an iterator to the beginning of the data that was already cuda memcopied onto the gpu
		thrust::device_ptr<popComplex> d_vec_begin = thrust::device_pointer_cast(in);

//		// transform between two vectors like this:
//		// http://thrust.github.io/doc/group__transformations.html#ga68a3ba7d332887f1332ca3bc04453792

		indexed_magnitude_squared_functor_fixed functor = indexed_magnitude_squared_functor_fixed();
		functor.m_len = len;

		// this function is weird because it takes begin1, end1, begin2 but not end2.  so therefore end2 is calculated based on begin/end 1
		// the 4th argument is the beginning of the output, and the 5th is the functor

		// takes about 42000us
		thrust::transform(
				thrust::make_counting_iterator(0),
				thrust::make_counting_iterator(totalLen),
				d_vec_begin,
				d_mag_vec->begin(),
				functor);


		// find the maximum element using the gpu, and return a pointer to it (a device_vector::iterator)
		// this takes about 36000us
		thrust::device_vector<double>::iterator d_max_element_itr = thrust::max_element(d_mag_vec->begin(), d_mag_vec->end());

		unsigned int position = d_max_element_itr - d_mag_vec->begin();
		double max_val = *d_max_element_itr;
		*peak = max_val;
	}



}
