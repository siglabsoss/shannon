/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// This define includes a main into our program for us
#define BOOST_TEST_DYN_LINK

// see http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/ for more info
#define BOOST_TEST_MODULE PopGeoLocationTest

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <vector>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "core/geohelper.hpp"
#include "core/popbuchermultilateration.hpp"
#include "core/popgeolocation.hpp"
#include "core/popsighting.hpp"

using pop::GeoHelper;
using pop::PopBucherMultilateration;
using pop::PopGeoLocation;
using pop::PopSighting;
using std::vector;

BOOST_AUTO_TEST_CASE(calculate_location)
{
	static const uint64_t TRACKER_ID = 13579;
	static const time_t FULL_SECS = 1400556041;

	GeoHelper geo_helper;
	PopBucherMultilateration multilateration(&geo_helper);
	PopGeoLocation geo_location(&geo_helper, &multilateration);

	vector<PopSighting> sightings;

	PopSighting sight;
	sight.tracker_id = TRACKER_ID;
	sight.full_secs = FULL_SECS;

	sight.hostname = "Denver";
	sight.lat = 39.7643389;
	sight.lng = -104.8551115;
	sight.frac_secs = 0.004223709183504543;
	sightings.push_back(sight);

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

	sight.hostname = "Calgary";
	sight.lat = 51.013117;
	sight.lng = -114.0741556;
	sight.frac_secs = 0.007719569969969025;
	sightings.push_back(sight);

	double lat = 0.0, lng = 0.0;
	geo_location.calculate_location(sightings, &lat, &lng);

	printf("lat == %f, lng == %f\n\n", lat, lng);

	// St. Louis
	BOOST_CHECK_CLOSE(lat,  38.6537065, 0.001);
	BOOST_CHECK_CLOSE(lng, -90.2477908, 0.001);
}
