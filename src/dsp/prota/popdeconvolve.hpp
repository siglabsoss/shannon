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
#include <mdl/popsymbol.hpp>

#include <cufft.h>

#include <thrust/device_vector.h>

#include <dsp/common/poptypes.h>

using namespace boost::posix_time;



#define SPREADING_LENGTH (4096)
#define SPREADING_BINS   (400)
#define SPREADING_CODES  (2)




namespace pop
{
	class PopProtADeconvolve : public PopSink<std::complex<double>[50] >
	{
	public:
		PopProtADeconvolve();
		~PopProtADeconvolve();
		PopSource<std::complex<double> > cts;
		PopSource<PopSymbol> maxima;

	private:
		void process(const std::complex<double> (*in)[50], size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void init();

		static void gpu_gen_pn_match_filter_coef(const int8_t* prn, std::complex<double>* cfc,
	                                      size_t  ncs, size_t osl, double bt);
		double sincInterpolateMaxima(const std::complex<double>* data, int sample, int neighbors );

	private:
		cufftHandle plan_fft;
		cufftHandle plan_deconvolve;
		cufftHandle many_plan_fft;
		popComplex* d_sts; // sampled time series
		popComplex* d_sfs; // sampled fourier series
		popComplex* d_cfs; // convoluted frequency swept series
		popComplex* d_cts; // convoluted time series
		thrust::device_vector<double>* d_mag_vec; // convoluted time series magnitude
		popComplex* d_cfc[SPREADING_LENGTH]; // convolution filter coefficients
		int*       d_peaks; // array of indices of detected peaks
		unsigned int*  	   d_peaks_len; // index of last detected peak
		std::complex<double>* d_sinc_yp; // samples for sinc interpolation around detected peak

		friend class blahtest;
	};
}

#endif // __POP_PROT_A_DECONVOLVE_H
