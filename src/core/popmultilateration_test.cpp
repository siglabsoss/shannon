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

void Test1()
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

void Test2()
{
	PopMultilateration multilateration;

	vector<PopSighting> sightings(3);

	// Los Angeles
	sightings[0].hostname = "hostname_a";
	sightings[0].tracker_id = 13579;
	sightings[0].lat = 34.0204989;
	sightings[0].lng = -118.4117325;
	sightings[0].full_secs = 1400556041;
	sightings[0].frac_secs = 0.006404773758517968;

	// Winnipeg
	sightings[1].hostname = "hostname_b";
	sightings[1].tracker_id = 13579;
	sightings[1].lat = 49.853822;
	sightings[1].lng = -97.1522251;
	sightings[1].full_secs = 1400556041;
	sightings[1].frac_secs = 0.004024396897936639;

	// Miami
	sightings[2].hostname = "hostname_c";
	sightings[2].tracker_id = 13579;
	sightings[2].lat = 25.782324;
	sightings[2].lng = -80.2310801;
	sightings[2].full_secs = 1400556041;
	sightings[2].frac_secs = 0.013372558268293727;

	double lat = 0.0, lng = 0.0;
	multilateration.calculate_location(sightings, &lat, &lng);

	// Should be "lat == 51.013117 , lng == -114.0741556" (Calgary).
	printf("lat == %f , lng == %f\n", lat, lng);
}

}

int main()
{
	Test2();
	return 0;
}
