/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BINNER_HPP__
#define __POP_BINNER_HPP__

#include <complex>
#include <cstring>
#include <stdint.h>

#include "core/popsource.hpp"
#include "core/popsink.hpp"
#include "core/popsinkgpu.hpp"
#include "mdl/popsymbol.hpp"
#include "mdl/poppeak.hpp"
#include "dsp/prota/popdeconvolve.hpp"
#include "core/basestationfreq.h"


#include <cufft.h>

#include <thrust/device_vector.h>

#include "dsp/common/poptypes.h"

using namespace boost::posix_time;




namespace pop
{
	class PopBinner : public PopSinkGpu<popComplex[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS] >
	{
	public:
		PopBinner();
		~PopBinner();

	private:
		void process(const popComplex (*data)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size);
		void init();

	private:
		cudaStream_t binner_stream;
		int*       d_peaks; // array of indices of detected peaks
		unsigned int*  	   d_peaks_len; // index of last detected peak
		int*       d_maxima_peaks; // array of indices of detected peaks that are local maxima
		unsigned int*  	   d_maxima_peaks_len; // index of last detected peak that is a local maxima
	};
}

#endif
