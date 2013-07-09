/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SDR_H
#define __POP_SDR_H

#include <cstring>
#include <complex>

#include <boost/thread.hpp>
#include <boost/signals2.hpp>
#include <uhd/usrp/multi_usrp.hpp>

#include "popassert.h"

#include <popsource.hpp>

namespace pop
{
	typedef void (*SDR_DATA_FUNC)(void* data, std::size_t len);

	class PopSdr : public PopSource<>
	{
	public:
		PopSdr();
		~PopSdr();
		POP_ERROR start();
		POP_ERROR stop();

	private:
		POP_ERROR run();
		uhd::usrp::multi_usrp::sptr usrp;
		uhd::rx_streamer::sptr rx_stream;
		uhd::tx_streamer::sptr tx_stream;
		uhd::rx_metadata_t md;
		boost::thread *mp_thread;
	};
}

#endif // __POP_SDR_H
