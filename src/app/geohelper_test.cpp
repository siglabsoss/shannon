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

#include <string>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"

using boost::tie;
using pop::GeoHelper;
using std::string;

BOOST_AUTO_TEST_CASE(turn_llh_into_xyz)
{
	GeoHelper geo_helper;

	const double lat_dec = 37.506794;
	const double long_dec = -122.204533;
	const double height = 0.0;
	const string coord_system = "wgs84";

	double x = 0.0, y = 0.0, z = 0.0;
	tie(x, y, z) = geo_helper.turn_llh_into_xyz(lat_dec, long_dec, height,
												coord_system);

	BOOST_CHECK_CLOSE(x, -2699861.0, 0.0001);
	BOOST_CHECK_CLOSE(y, -4286555.0, 0.0001);
	BOOST_CHECK_CLOSE(z,  3862162.0, 0.0001);
}
