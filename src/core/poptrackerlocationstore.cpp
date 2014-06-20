/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "core/poptrackerlocationstore.hpp"
#include "mdl/popradio.h"
#include "mdl/poptimestamp.hpp"
#include "net/popwebhook.hpp"

namespace pop
{

PopTrackerLocationStore::PopTrackerLocationStore(PopWebhook* webhook)
	: webhook_(webhook)
{
	assert(webhook != NULL);
}

void PopTrackerLocationStore::report_tracker_location(uint64_t tracker_id,
													  time_t full_secs,
													  double lat, double lng)
{
	printf("tracker_id == %" PRIu64 ", full_secs == %" PRId64 ", lat == %0.8f, "
		   "lng == %0.8f\n",
		   tracker_id, full_secs, lat, lng);

	// Send the tracker location to the web hook.
	PopRadio radio;
	radio.setLat(lat);
	radio.setLng(lng);
	radio.setSerial(static_cast<long>(tracker_id));

	// TODO(snyderek): Include the fractional seconds in the timestamp.
	const PopTimestamp timestamp(full_secs, 0.0);

	webhook_->process(&radio, 1, &timestamp, 1);
}

}
