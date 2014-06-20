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
#define BOOST_TEST_MODULE PopBucherMultilaterationTest

#include <vector>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"
#include "core/popbuchermultilateration.hpp"

using boost::make_tuple;
using boost::tie;
using boost::tuple;
using pop::GeoHelper;
using pop::PopBucherMultilateration;
using std::vector;

BOOST_AUTO_TEST_CASE(calculate_xyz)
{
	GeoHelper geo_helper;
	PopBucherMultilateration multilateration(&geo_helper);

	vector<tuple<double, double, double, double> > sets(4);
	sets[0] = make_tuple(-0.0041984789346775, -0.0158289403417070, 0.0135357198108165, 0.0042167957666482);
	sets[1] = make_tuple(-0.0083989773737108, -0.0155259839007040, 0.0118359595092561, 0.0085134727938702);
	sets[2] = make_tuple(-0.0017111922827391, -0.0136369181952944, 0.0161856041543917, 0.0045271098298357);
	sets[3] = make_tuple( 0.0032525731902556, -0.0188914824983019, 0.0091975863743956, 0.0056828758960522);

	tuple<double, double, double> result;
	BOOST_CHECK(multilateration.calculate_xyz(sets, &result));

	double x, y, z;
	tie(x, y, z) = result;

	BOOST_CHECK_CLOSE(x, -0.0000719464396933, 0.000000001);
	BOOST_CHECK_CLOSE(y, -0.0166361189780165, 0.000000001);
	BOOST_CHECK_CLOSE(z,  0.0132170369676211, 0.000000001);
}

// Same as the previous test, but swap the order of the last two base stations.
BOOST_AUTO_TEST_CASE(swap_base_stations_2_and_3)
{
	GeoHelper geo_helper;
	PopBucherMultilateration multilateration(&geo_helper);

	vector<tuple<double, double, double, double> > sets(4);
	sets[0] = make_tuple(-0.0041984789346775, -0.0158289403417070, 0.0135357198108165, 0.0042167957666482);
	sets[1] = make_tuple(-0.0083989773737108, -0.0155259839007040, 0.0118359595092561, 0.0085134727938702);
	sets[2] = make_tuple( 0.0032525731902556, -0.0188914824983019, 0.0091975863743956, 0.0056828758960522);
	sets[3] = make_tuple(-0.0017111922827391, -0.0136369181952944, 0.0161856041543917, 0.0045271098298357);

	tuple<double, double, double> result;
	BOOST_CHECK(multilateration.calculate_xyz(sets, &result));

	double x, y, z;
	tie(x, y, z) = result;

	BOOST_CHECK_CLOSE(x, -0.0000719464396933, 0.000000001);
	BOOST_CHECK_CLOSE(y, -0.0166361189780165, 0.000000001);
	BOOST_CHECK_CLOSE(z,  0.0132170369676211, 0.000000001);
}

BOOST_AUTO_TEST_CASE(calculate_xyz_2)
{
	GeoHelper geo_helper;
	PopBucherMultilateration multilateration(&geo_helper);

	vector<tuple<double, double, double, double> > sets(4);
	sets[0] = make_tuple( 0.0013788593431586,  0.0031199536615263, -0.0209295365457260, 0.0209609386560155);
	sets[1] = make_tuple( 0.0116576097323980, -0.0003087582448382, -0.0177346604029982, 0.0106821026204624);
	sets[2] = make_tuple(-0.0077420756009415,  0.0070581067266477, -0.0184548400063539, 0.0292632721425059);
	sets[3] = make_tuple( 0.0088183984359118, -0.0143610099184906,  0.0129422346579387, 0.0264320202554581);

	tuple<double, double, double> result;
	BOOST_CHECK(multilateration.calculate_xyz(sets, &result));

	double x, y, z;
	tie(x, y, z) = result;

	BOOST_CHECK_CLOSE(x,  0.0170980990127883, 0.000000001);
	BOOST_CHECK_CLOSE(y, -0.0064255140532690, 0.000000001);
	BOOST_CHECK_CLOSE(z, -0.0108721760741216, 0.000000001);
}
