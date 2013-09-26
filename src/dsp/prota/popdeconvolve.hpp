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
#include <core/popsink.hpp>

#include <cufft.h>

using namespace boost::posix_time;

namespace pop
{
	class PopProtADeconvolve : public PopSink<std::complex<float> >, public PopSource<std::complex<float> >
	{
	public:
		PopProtADeconvolve();
		~PopProtADeconvolve();

	private:
		void process(const std::complex<float>* in, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void init();

		static void gpu_gen_pn_match_filter_coef(const int8_t* prn, std::complex<float>* cfc,
	                                      size_t  ncs, size_t osl, float bt);

	private:
		cufftHandle plan_fft;
		cufftHandle plan_deconvolve;
		cuComplex* d_sts; // sampled time series
		cuComplex* d_sfs; // sampled fourier series
		cuComplex* d_cfs; // convoluted frequency swept series
		cuComplex* d_cts; // convoluted time series
		cuComplex* d_cfc; // convolution filter coefficients
		float* d_peak; // detector output

		friend class blahtest;
	};
}

#endif // __POP_PROT_A_DECONVOLVE_H
