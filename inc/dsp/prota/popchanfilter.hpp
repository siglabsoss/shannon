/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_PROT_A_DESPREAD_H
#define __POP_PROT_A_DESPREAD_H

#include <complex>
#include <cstring>
#include <boost/thread.hpp>
#include <boost/signals2.hpp>
#include <boost/timer.hpp>

#include "core/popblock.hpp"

using namespace boost::posix_time;

namespace pop
{
	class PopChanFilter : public PopSink<std::complex<double> >, public PopSource<std::complex<double> >
	{
	public:
		PopChanFilter();
		~PopChanFilter();
		static double fbin_size();
		static double fft_bottom_frequency();
		static double fft_top_frequency();
		static double fbins_per_channel();
		static double channel_frequency(unsigned c);
		static double channel_frequency_above_fft(unsigned c);
		static double channel_fbin_center(unsigned c);
		static double channel_fbin_low_exact(unsigned c);
		static double channel_fbin_high_exact(unsigned c);
		static size_t channel_fbin_low(unsigned c);
		static size_t channel_fbin_high(unsigned c);

		PopSource<std::complex<double>[50]> strided;

	private:
		void process(const std::complex<double>* in, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void init();

		void gen_dft(std::complex<double>* out, size_t bins);
		void gen_inv_dft(std::complex<double>* out, size_t bins);
		void gen_carrier(std::complex<double>* out, size_t bins, size_t harmonic);

		std::complex<double> *mp_demod_func; ///< PN code vector
	};
}

#endif // __POP_PROT_A_DESPREAD_H
