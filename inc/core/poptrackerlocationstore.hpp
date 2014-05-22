/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_TRACKER_LOCATION_STORE__
#define __POP_TRACKER_LOCATION_STORE__

#include <stdint.h>
#include <time.h>

namespace pop
{

// In-memory store of tracker locations that have been computed by
// multilateration. The actual computation is performed in the
// PopMultilateration class.
class PopTrackerLocationStore
{
public:
	PopTrackerLocationStore();

	void report_device_location(uint64_t tracker_id, time_t full_secs,
								double lat, double lng);

private:
};

}

#endif
