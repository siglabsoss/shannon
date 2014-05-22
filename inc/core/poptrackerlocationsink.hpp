/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_TRACKER_LOCATION_SINK__
#define __POP_TRACKER_LOCATION_SINK__

#include <stdint.h>
#include <time.h>

namespace pop
{

class PopTrackerLocationSink
{
public:
	PopTrackerLocationSink();

	void report_device_location(uint64_t tracker_id, time_t full_secs,
								double lat, double lng);

private:
};

}

#endif
