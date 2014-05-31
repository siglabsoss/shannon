/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using boost::make_tuple;
using boost::tie;
using boost::tuple;
using pop::PopMultilateration;
using pop::PopSighting;
using pop::calculate_xyz;
using std::vector;

namespace {

void test2()
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

	// Should be "lat == 38.6537065 , lng == -90.2477908" (St. Louis).
	printf("lat == %f , lng == %f\n", lat, lng);
}

void test3()
{
	vector<tuple<double, double, double, double> > sets(5);
	sets[0] = make_tuple(-1258672.3196881897747517, -4745396.9325757101178169, 4057906.7128839851357043, 0.0042167957666482);
	sets[1] = make_tuple(-2517950.0715511469170451, -4654572.8764604805037379, 3548331.3940683654509485, 0.0085134727938702);
	sets[2] = make_tuple( -513002.5405529749114066, -4088245.2253122413530946, 4852322.0536601003259420, 0.0045271098298356);
	sets[3] = make_tuple(  975096.9115316333482042, -5663523.9734298968687654, 2757367.0268473550677299, 0.0056828758960522);
	sets[4] = make_tuple(-1640198.7441209338139743, -3671151.9755489402450621, 4934462.8230686001479626, 0.0076773345072259);

	double x, y, z;
	tie(x, y, z) = calculate_xyz(sets);

	// Should be "x == -21569.0, y == -4987383.0, z == 3962368.0".
	printf("x == %f, y == %f, z == %f\n", x, y, z);
}

}

int main()
{
	test3();
	return 0;
}
