/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <stdio.h>

#include <vector>

#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using pop::PopMultilateration;
using pop::PopSighting;
using std::vector;

namespace {

/*
void test1()
{
	PopMultilateration multilateration;

	vector<PopSighting> sightings(3);

	sightings[0].hostname = "hostname_a";
	sightings[0].tracker_id = 13579;
	sightings[0].lat = 37.506794;
	sightings[0].lng = -122.204533;
	sightings[0].full_secs = 1400556041;
	sightings[0].frac_secs = 0.000011391297242040692;

	sightings[1].hostname = "hostname_b";
	sightings[1].tracker_id = 13579;
	sightings[1].lat = 37.471107;
	sightings[1].lng = -122.235775;
	sightings[1].full_secs = 1400556041;
	sightings[1].frac_secs = 0.000011081226066067346;

	sightings[2].hostname = "hostname_c";
	sightings[2].tracker_id = 13579;
	sightings[2].lat = 37.440583;
	sightings[2].lng = -122.142735;
	sightings[2].full_secs = 1400556041;
	sightings[2].frac_secs = 0.000021196070249372317;

	double lat = 0.0, lng = 0.0;
	multilateration.calculate_location(sightings, &lat, &lng);

	// Should be "lat == 37.476365 , lng == -122.198803".
	printf("lat == %f , lng == %f\n", lat, lng);
}
*/

void test2()
{
	static const uint64_t TRACKER_ID = 13579;
	static const time_t FULL_SECS = 1400556041;

	PopMultilateration multilateration;

	vector<PopSighting> sightings;

	PopSighting sight;
	sight.tracker_id = TRACKER_ID;
	sight.full_secs = FULL_SECS;

	sight.hostname = "Los Angeles";
	sight.lat = 34.0204989;
	sight.lng = -118.4117325;
	sight.frac_secs = 0.008571202011359472;
	sightings.push_back(sight);

	sight.hostname = "Winnipeg";
	sight.lat = 49.853822;
	sight.lng = -97.1522251;
	sight.frac_secs = 0.004535715127963626;
	sightings.push_back(sight);

	sight.hostname = "Miami";
	sight.lat = 25.782324;
	sight.lng = -80.2310801;
	sight.frac_secs = 0.005699991829013924;
	sightings.push_back(sight);

	sight.hostname = "Denver";
	sight.lat = 39.7643389;
	sight.lng = -104.8551115;
	sight.frac_secs = 0.004223709183504543;
	sightings.push_back(sight);

	double lat = 0.0, lng = 0.0;
	multilateration.calculate_location(sightings, &lat, &lng);

	// Should be "lat == 51.013117 , lng == -114.0741556" (Calgary).
	printf("lat == %f , lng == %f\n", lat, lng);
}

}

int main()
{
	test2();
	return 0;
}
