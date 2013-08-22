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

#include "core/global.h"
#include "core/popassert.h"
#include "core/popsource.hpp"

namespace pop
{

#define POP_PROTA_BLOCK_A_UPLK 903626953
#define POP_PROTA_BLOCK_B_UPLK 906673828
#define POP_PROTA_BLOCK_C_UPLK 909720703
#define POP_PROTA_BLOCK_D_UPLK 912767578

#define POP_PROTA_BLOCK_A_DOWN 917236328
#define POP_PROTA_BLOCK_B_DOWN 920283203
#define POP_PROTA_BLOCK_C_DOWN 923330078
#define POP_PROTA_BLOCK_D_DOWN 926376953

#define POP_PROTA_BLOCK_A_WIDTH 3200000
#define POP_PROTA_BLOCK_B_WIDTH 3200000
#define POP_PROTA_BLOCK_C_WIDTH 3200000
#define POP_PROTA_BLOCK_D_WIDTH 3200000

	typedef void (*SDR_DATA_FUNC)(void* data, std::size_t len);

	class PopUhd : public PopSource<>
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
	};
}

#endif // __POP_UHD_H
