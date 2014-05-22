/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "core/poptrackerlocationstore.hpp"

namespace pop
{

PopTrackerLocationStore::PopTrackerLocationStore()
{
}

void PopTrackerLocationStore::report_device_location(uint64_t tracker_id,
													 time_t full_secs,
													 double lat, double lng)
{
	printf("tracker_id == %" PRIu64 ", full_secs == %" PRId64 ", lat == %0.8f, "
		   "lng == %0.8f\n",
		   tracker_id, full_secs, lat, lng);
}

}
