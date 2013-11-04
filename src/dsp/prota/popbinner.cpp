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


#include "dsp/prota/popbinner.hpp"
#include "core/basestationfreq.h"



using namespace std;
using namespace boost::posix_time;

namespace pop
{


PopBinner::PopBinner() : PopSinkGpu<popComplex[50][SPREADING_CODES][SPREADING_BINS]>( "PopBinner", SPREADING_LENGTH*2 )
{

}

PopBinner::~PopBinner()
{
	cudaStreamDestroy(binner_stream);
}



void PopBinner::init()
{
    // create CUDA stream
    checkCudaErrors(cudaStreamCreate(&binner_stream));
}






//// assumes input has not been shifted and is in the form of [m -> h][l -> m] where the center half is bad data
//// assumes spreadLength to be 2x the number of valid samples
//// returns a pair ( time , bin )
//boost::tuple<double, int> PopBinner::linearToBins(double sample, int spreadLength, int fbins)
//{
//	double timeIndex;
//	int fbin;
//
//	// add half of spread length to offset us to the center, then mod
//	// this leaves us with [l -> m][m -> h] where the first and last quarters are bad data
//	timeIndex = fmod( sample + (spreadLength/2), spreadLength );
//
//	// subtract 1 quarter to shift the good data into the first half.
//	// this ranges the timeIndex from (0 - spreadLength/2)
//	timeIndex -= (spreadLength / 4);
//
//	fbin = floor(sample / spreadLength);
//
//	return boost::tuple<double,int>(timeIndex,fbin);
//}
//



void PopBinner::process(const popComplex(*data)[50][SPREADING_CODES][SPREADING_BINS], size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{

	cout << "in PopBinner with " << data_size << " samples and " << timestamp_size << " stamps " << endl;

//	// this is a running counter which represents the x-offset of the first valid sample (without padding) that will be unique for a long time
//	static size_t running_counter = 0;


//	ptime t1, t2;
//	time_duration td, tLast;
//	t1 = microsec_clock::local_time();

	static popComplex h_cts[50][SPREADING_CODES][SPREADING_BINS][SPREADING_LENGTH*2];

	cudaMemcpy(h_cts, data, 50 * SPREADING_CODES * SPREADING_BINS * SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyDeviceToHost);

	std::complex<double> *val;

	for( size_t i = 0; i < SPREADING_LENGTH * 2; i++ )
	{
		val = (std::complex<double>*) &(h_cts[9][0][0][i]);
		cout << "  " << *val << endl;
	}

	int j = 666666;



	// copy new host data into device memory
//	cudaMemcpyAsync(d_sts, in - SPREADING_LENGTH, 50 * SPREADING_LENGTH * 2 * sizeof(popComplex), cudaMemcpyHostToDevice, deconvolve_stream);
//	cudaThreadSynchronize();

//	// perform FFTs on each channel (50)
//	cufftExecZ2Z(many_plan_fft, (cufftDoubleComplex*)(in - SPREADING_LENGTH), (cufftDoubleComplex*)d_sfs, CUFFT_FORWARD);
////	cudaThreadSynchronize();
//
//
//	for( int channel = 0; channel < 14; channel++ )
//	{
//		deconvolve_channel(bsf_channel_sequence[channel], running_counter, in, len, timestamp_data, timestamp_size);
//	}
//
//	running_counter += SPREADING_LENGTH;

//	t2 = microsec_clock::local_time();
//	td = t2 - t1;
	//cout << " PopDeconvolve - 1040 RF samples received and computed in " << td.total_microseconds() << "us." << endl;

}


} // namespace pop

