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
#define BOOST_TEST_MODULE PopCoordinateTransformTest

#include <math.h>

#include <boost/test/auto_unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/popcoordinatetransform.hpp"

using boost::get;
using boost::make_tuple;
using boost::tie;
using boost::tuple;
using pop::PopCoordinateTransform;

namespace
{

double sqr(double x)
{
	return x * x;
}

double distance(const tuple<double, double, double>& a,
				const tuple<double, double, double>& b)
{
	return sqrt(sqr(get<0>(a) - get<0>(b)) +
				sqr(get<1>(a) - get<1>(b)) +
				sqr(get<2>(a) - get<2>(b)));
}

}  // namespace

BOOST_AUTO_TEST_CASE(base_stations)
{
	static const double kTolerance = 0.0001;

	const tuple<double, double, double> base_station1(
		-0.0041984789346775, -0.0158289403417070, 0.0135357198108165);
	const tuple<double, double, double> base_station2(
		-0.0083989773737108, -0.0155259839007040, 0.0118359595092561);

	const double dist = distance(base_station1, base_station2);

	const PopCoordinateTransform coordinate_transform(base_station1,
													  base_station2);

	double x, y, z;
	tie(x, y, z) = coordinate_transform.transform(base_station1);
	// BOOST_CHECK_CLOSE doesn't work well for comparing floating-point values
	// against zero, because the tolerance parameter is interpreted as a
	// percentage of the values being compared, and any percentage multiplied by
	// zero is going to be zero.
	BOOST_CHECK_SMALL(x, kTolerance);
	BOOST_CHECK_SMALL(y, kTolerance);
	BOOST_CHECK_SMALL(z, kTolerance);

	tie(x, y, z) = coordinate_transform.transform(base_station2);
	BOOST_CHECK_CLOSE(x, dist, kTolerance);
	BOOST_CHECK_SMALL(y, kTolerance);
	BOOST_CHECK_SMALL(z, kTolerance);

	tie(x, y, z) = coordinate_transform.untransform(make_tuple(0.0, 0.0, 0.0));
	BOOST_CHECK_CLOSE(x, get<0>(base_station1), kTolerance);
	BOOST_CHECK_CLOSE(y, get<1>(base_station1), kTolerance);
	BOOST_CHECK_CLOSE(z, get<2>(base_station1), kTolerance);

	tie(x, y, z) = coordinate_transform.untransform(make_tuple(dist, 0.0, 0.0));
	BOOST_CHECK_CLOSE(x, get<0>(base_station2), kTolerance);
	BOOST_CHECK_CLOSE(y, get<1>(base_station2), kTolerance);
	BOOST_CHECK_CLOSE(z, get<2>(base_station2), kTolerance);
}
