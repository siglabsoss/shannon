/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_PROT_A_DECONVOLVE_H
#define __POP_PROT_A_DECONVOLVE_H

#include <complex>
#include <cstring>
#include <stdint.h>

#include <core/popsource.hpp>
#include <core/popsourcegpu.hpp>
#include <core/popsink.hpp>
#include <core/popsinkgpu.hpp>
#include <mdl/popsymbol.hpp>
#include <mdl/poppeak.hpp>


#include <cufft.h>

#include <thrust/device_vector.h>

#include <dsp/common/poptypes.h>
#include "core/basestationfreq.h"

using namespace boost::posix_time;



#define SPREADING_LENGTH (512)
#define SPREADING_BINS   (100)
#define SPREADING_CODES  (2)       //  some code assumes the value of this to be 2




namespace pop
{
	class PopProtADeconvolve : public PopSinkGpu<std::complex<double>[CHANNELS_USED] >
	{
	public:
		PopProtADeconvolve();
		~PopProtADeconvolve();
		PopSource<std::complex<double> > cts;
		PopSource<PopSymbol> maxima;
		PopSource<PopPeak> peaks;
		PopSourceGpu<double[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS]> cts_mag_gpu;

	private:
		void process(const std::complex<double> (*in)[CHANNELS_USED], size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void deconvolve_channel(unsigned channel, size_t running_counter, double (*cts_stream_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], const std::complex<double> (*in)[CHANNELS_USED], size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void init();

		static void gpu_gen_pn_match_filter_coef(const int8_t* prn, std::complex<double>* cfc,
	                                      size_t  ncs, size_t osl, double bt);
		double sincInterpolateMaxima(popComplex* data, int sample, int neighbors );

	private:
		cufftHandle plan_fft;
		cufftHandle plan_deconvolve;
		cufftHandle many_plan_fft;
		cudaStream_t deconvolve_stream;
		popComplex* d_sts; // sampled time series
		popComplex* d_sfs; // sampled fourier series
		popComplex* d_cfs; // convoluted frequency swept series
		popComplex* d_cts; // convoluted time series
		popComplex* d_cfc[SPREADING_LENGTH]; // convolution filter coefficients
		std::complex<double>* d_sinc_yp; // samples for sinc interpolation around detected peak

		friend class blahtest;
	};
}

#endif // __POP_PROT_A_DECONVOLVE_H
