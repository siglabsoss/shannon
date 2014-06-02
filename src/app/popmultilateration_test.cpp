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
#define BOOST_TEST_MODULE GeoHelperTest

#include <stdint.h>
#include <time.h>

#include <vector>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using boost::make_tuple;
using boost::tie;
using boost::tuple;
using pop::PopMultilateration;
using pop::PopSighting;
using std::vector;

BOOST_AUTO_TEST_CASE(calculate_location)
{
	static const uint64_t TRACKER_ID = 13579;
	static const time_t FULL_SECS = 1400556041;

	PopMultilateration multilateration;

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
	multilateration.calculate_location(sightings, &lat, &lng);

	// St. Louis
	BOOST_CHECK_CLOSE(lat,  38.6537065, 0.0001);
	BOOST_CHECK_CLOSE(lng, -90.2477908, 0.0001);
}

BOOST_AUTO_TEST_CASE(calculate_xyz)
{
	vector<tuple<double, double, double, double> > sets(5);
	sets[0] = make_tuple(-0.0041984789346775, -0.0158289403417070, 0.0135357198108165, 0.0042167957666482);
	sets[1] = make_tuple(-0.0083989773737108, -0.0155259839007040, 0.0118359595092561, 0.0085134727938702);
	sets[2] = make_tuple(-0.0017111922827391, -0.0136369181952944, 0.0161856041543917, 0.0045271098298357);
	sets[3] = make_tuple( 0.0032525731902556, -0.0188914824983019, 0.0091975863743956, 0.0056828758960522);
	sets[4] = make_tuple(-0.0054711141002784, -0.0122456448705889, 0.0164595962686580, 0.0076773345072259);

	double x, y, z;
	tie(x, y, z) = pop::calculate_xyz(sets);

	BOOST_CHECK_CLOSE(x, -0.0000719464396933, 0.000000001);
	BOOST_CHECK_CLOSE(y, -0.0166361189780165, 0.000000001);
	BOOST_CHECK_CLOSE(z,  0.0132170369676211, 0.000000001);
}
