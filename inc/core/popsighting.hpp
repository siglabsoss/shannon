/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SIGHTING__
#define __POP_SIGHTING__

#include <stdint.h>
#include <time.h>

#include <string>

namespace pop
{

struct PopSighting
{
	// Host name of the base station.
	std::string hostname;

	// Unique ID from the tracker board.
	// TODO(snyderek): What is the correct data type?
	uint64_t tracker_id;

	// Latitude and longitude
	double lat;
	double lng;

	// Integer component of the timestamp when the tracker signal was received
	// by the base station. (Seconds since the epoch.)
	time_t full_secs;

	// Fractional component of the timestamp when the tracker signal was
	// received by the base station.
	double frac_secs;
};

}

#endif
