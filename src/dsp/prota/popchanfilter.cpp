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

#include "cuda/helper_cuda.h"

#include "dsp/prota/popchanfilter.hpp"

using namespace std;
using namespace boost::posix_time;

#define FFT_SIZE 65536
#define CHAN_SIZE 1040


/**************************************************************************
 * CUDA Function Prototypes
 *************************************************************************/
extern "C" size_t gpu_channel_split(const complex<float> *h_data, complex<float> *out);
extern "C" void init_deconvolve(size_t len_fft, size_t len_chan);
extern "C" void cleanup();

namespace pop
{

	PopChanFilter::PopChanFilter(): PopSink<complex<float> >( "PopChanFilter", FFT_SIZE ),
		PopSource<complex<float> >( "PopChanFilter" ), mp_demod_func(0)
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
	void PopChanFilter::process(const complex<float>* in, size_t len)
	{
		//size_t chan_buf_len;
		ptime t1, t2;
		time_duration td, tLast;
		t1 = microsec_clock::local_time();

		complex<float> *out = get_buffer(CHAN_SIZE);

		//cudaProfilerStart();
		// call the GPU to process work
		gpu_channel_split(in, out);

		// process data
		PopSource<complex<float> >::process();

		// while( chan_buf_len >= PN_SIZE)
		// {
		// 	complex<float> *out = get_buffer(PN_SIZE);
		// 	chan_buf_len = gpu_demod(out);
		// 	PopSource<complex<float> >::process();
		// }
		
		
		//cudaProfilerStop();

		t2 = microsec_clock::local_time();
		td = t2 - t1;

		//cout << PopSource<complex<float> >::get_name() << " - 65536 RF samples received and computed in " << td.total_microseconds() << "us." << endl;
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
}
