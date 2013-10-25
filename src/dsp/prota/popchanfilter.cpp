/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <memory>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <boost/timer.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <boost/lexical_cast.hpp>
#include "sdr/popuhd.hpp"

#include "cuda/helper_cuda.h"

#include "dsp/prota/popchanfilter.hpp"

using namespace std;
using namespace boost::posix_time;

#define FFT_SIZE 65536
#define CHAN_SIZE 1040
#define POP_CHANNEL_SPACING ((double) 50781.25)
#define POP_PROTA_BLOCK_A_CHANNEL_0 ((double) 902382812.50)


/**************************************************************************
 * CUDA Function Prototypes
 *************************************************************************/
extern "C" size_t gpu_channel_split(const complex<double> *h_data, complex<double> *out);
extern "C" void init_deconvolve(size_t len_fft, size_t len_chan);
extern "C" void cleanup();

namespace pop
{

	PopChanFilter::PopChanFilter(): PopSink<complex<double> >( "PopChanFilter", FFT_SIZE ),
		PopSource<complex<double> >( "PopChanFilter" ), mp_demod_func(0)
	{
	}

	void PopChanFilter::init()
	{
    	// Init CUDA
	 	int deviceCount = 0;
	    cout << "initializing graphics card(s)...." << endl;
	    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	    if (error_id != cudaSuccess)
	    {
	        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	        exit(EXIT_FAILURE);
	    }

	    // This function call returns 0 if there are no CUDA capable devices.
	    if (deviceCount == 0)
	    {
	        printf("There are no available device(s) that support CUDA\n");
	    }
	    else
	    {
	        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	    }

	    // choose which device to use for this thread
	    cudaSetDevice(0);
	    
	    // allocate CUDA memory
	    init_deconvolve( FFT_SIZE, CHAN_SIZE );
	}


	/**
	 * Process data.
	 */
	void PopChanFilter::process(const complex<double>* in, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		//size_t chan_buf_len;
		ptime t1, t2;
		time_duration td, tLast;
		t1 = microsec_clock::local_time();

		complex<double> *out = get_buffer(CHAN_SIZE);

//		//cudaProfilerStart();
//		// call the GPU to process work
		gpu_channel_split(in, out);


		// in comes FFT_SIZE (65K) samples, and out goes CHAN_SIZE (1040)
		// these 1040 samples still represent the same timestamps
		// so we need to just do a simple divide

		// get timestamp buffer
		PopTimestamp *timestampOut = get_timestamp_buffer(timestamp_size);

		// get correction factor (note we do (slower) division here outside of loop and then (faster) multiplication inside the loop
		double factor = (double) FFT_SIZE / CHAN_SIZE;

		double a;
//		double secs;

		// index of two timestamps to interpolate between
		size_t i1, i2;

		// two timestamps
		PopTimestamp ts1, ts2;

		for(size_t m = 0; m < CHAN_SIZE; m++ )
		{
			a = fmod( factor * m, 1.0 );

			i1 = floor(factor * m);
			i2 =  ceil(factor * m);

			ts1 = timestamp_data[i1];
			ts2 = timestamp_data[i2];

			ts1 *= (1-a);
			ts2 *= (a);

			// final result is stored in ts1
			ts1 += ts2;

			timestampOut[m] = ts1;
		}


		// process data
		PopSource<complex<double> >::process(out, CHAN_SIZE, timestampOut, CHAN_SIZE);

		// while( chan_buf_len >= PN_SIZE)
		// {
		// 	complex<double> *out = get_buffer(PN_SIZE);
		// 	chan_buf_len = gpu_demod(out);
		// 	PopSource<complex<double> >::process();
		// }
		
		
		//cudaProfilerStop();

		t2 = microsec_clock::local_time();
		td = t2 - t1;

		//cout << PopSource<complex<double> >::get_name() << " - 65536 RF samples received and computed in " << td.total_microseconds() << "us." << endl;
	}


	 /**
	  * Standard class deconstructor.
	  */
	PopChanFilter::~PopChanFilter()
	{
		if( mp_demod_func )
			free(mp_demod_func);

		// free CUDA memory
		cleanup();
	}

	double PopChanFilter::fbin_size()
	{
		return 1.0 / ((double)FFT_SIZE / POP_PROTA_BLOCK_A_WIDTH);
	}

	double PopChanFilter::fft_bottom_frequency()
	{
		return POP_PROTA_BLOCK_A_UPLK - (double) POP_PROTA_BLOCK_A_WIDTH / 2;
	}

	double PopChanFilter::fft_top_frequency()
	{
		return POP_PROTA_BLOCK_A_UPLK + (double) POP_PROTA_BLOCK_A_WIDTH / 2;
	}

	double PopChanFilter::fbins_per_channel()
	{
		return POP_CHANNEL_SPACING / fbin_size();
	}

	double PopChanFilter::channel_frequency(unsigned c)
	{
		return POP_PROTA_BLOCK_A_CHANNEL_0 + c * POP_CHANNEL_SPACING;
	}

	double PopChanFilter::channel_frequency_above_fft(unsigned c)
	{
		return channel_frequency(c) - fft_bottom_frequency();
	}

	double PopChanFilter::channel_fbin_center(unsigned c)
	{
		return channel_frequency_above_fft(c) / fbin_size();
	}

	double PopChanFilter::channel_fbin_low_exact(unsigned c)
	{
		return channel_fbin_center(c) - fbins_per_channel() / 2;
	}

	double PopChanFilter::channel_fbin_high_exact(unsigned c)
	{
		return channel_fbin_center(c) + fbins_per_channel() / 2;
	}

	// rounded versions of the above
	size_t PopChanFilter::channel_fbin_low(unsigned c)
	{
		return round( channel_fbin_low_exact(c) );
	}

	size_t PopChanFilter::channel_fbin_high(unsigned c)
	{
		return round( channel_fbin_high_exact(c) );
	}


















}
