/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"
#include "core/multilateration.hpp"
#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using boost::make_tuple;
using boost::tie;
using boost::tuple;
using std::vector;

namespace
{

const double SPEED_OF_LIGHT_M_PER_S = 299792458.0;

double sqr(double x)
{
	return x * x;
}

tuple<double, double, double> translate_xyz(double x, double y, double z,
											double dx, double dy, double dz)
{
	return make_tuple(x+dx, y+dy, z+dz);
}

tuple<double, double, double> rotate_xyz_around_z(double x, double y, double z,
												  double theta)
{
	return make_tuple(cos(theta)*x - sin(theta)*y,
					  sin(theta)*x + cos(theta)*y,
					  z);
}

tuple<double, double, double> rotate_xyz_around_x(double x, double y, double z,
												  double theta)
{
	return make_tuple(x,
					  cos(theta)*y - sin(theta)*z,
					  sin(theta)*y + cos(theta)*z);
}

tuple<double, double, double> rotate_xyz_around_y(double x, double y, double z,
												  double theta)
{
	return make_tuple(sin(theta)*z + cos(theta)*x,
					  y,
					  cos(theta)*z - sin(theta)*x);
}

double from_time_to_distance(double t)
{
	return SPEED_OF_LIGHT_M_PER_S * t;
}

double from_distance_to_time(double d)
{
	return d / SPEED_OF_LIGHT_M_PER_S;
}

}

namespace pop
{

PopMultilateration::PopMultilateration()
{
}

void PopMultilateration::calculate_location(
	const vector<PopSighting>& sightings, double* lat, double* lng) const
{
	assert(sightings.size() >= 4u);
	assert(lat != NULL);
	assert(lng != NULL);

	// Convert all the sightings from lat/long to (x,y,z) coordinates. For now,
	// only use the first four sightings in the computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	vector<tuple<double, double, double, double> > sets(4);

	for (vector<tuple<double, double, double, double> >::size_type i = 0;
		 i < sets.size(); ++i) {
		const PopSighting& sighting = sightings[i];

		// For now, assume that all base stations are at altitude 0.
		// TODO(snyderek): Should the base stations report their altitudes in
		// addition to lat/long?
		double x, y, z;
		tie(x, y, z) = geo_helper_.turn_llh_into_xyz(sighting.lat, sighting.lng,
													 0.0, "wgs84");

		sets[i] = make_tuple(x, y, z, sighting.frac_secs);
	}

	// Do the multilateration.
	double x, y, z;
	tie(x, y, z) = calculate_xyz_position(sets);

	double temp_lat, temp_lng, temp_alt;
	tie(temp_lat, temp_lng, temp_alt) = geo_helper_.turn_xyz_into_llh(x, y, z,
																	  "wgs84");

	*lat = temp_lat;
	*lng = temp_lng;
}

}
