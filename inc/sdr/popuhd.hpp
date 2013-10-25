/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_UHD_H
#define __POP_UHD_H

#include <cstring>
#include <complex>

#include <boost/thread.hpp>
#include <boost/signals2.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/types/time_spec.hpp>

#include "core/global.h"
#include "core/popassert.h"
#include "core/popsource.hpp"

// include macros for frequency constants
#include "core/basestationfreq.h"

namespace pop
{


	typedef void (*SDR_DATA_FUNC)(void* data, std::size_t len);

	class PopUhd : public PopSource<std::complex<double> >
	{
	public:
		PopUhd();
		~PopUhd();
		POP_ERROR start();
		POP_ERROR stop();

	private:
		POP_ERROR run();
		uhd::usrp::multi_usrp::sptr usrp;
		uhd::rx_streamer::sptr rx_stream;
		uhd::tx_streamer::sptr tx_stream;
		uhd::rx_metadata_t md;
		boost::thread *mp_thread;
		uhd::time_spec_t m_timestamp_offset;
	};
}

#endif // __POP_UHD_H
