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

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"
#include "core/popgeolocation.hpp"
#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using boost::make_tuple;
using boost::tie;
using boost::tuple;
using std::vector;

namespace pop
{



// Given a distance measured along a great circle between two points on the
// Earth's surface, returns the straight-line distance between the two points in
// Euclidean space. This function assumes that the Earth is spherical.
double PopGeoLocation::spherical_distance_to_linear(double dist_light_seconds)
{
	static const double EARTH_DIAMETER_LIGHT_SECONDS =
		GeoHelper::EARTH_RADIUS_M * 2.0 / GeoHelper::SPEED_OF_LIGHT_M_PER_S;

	return sin(dist_light_seconds / EARTH_DIAMETER_LIGHT_SECONDS) *
		EARTH_DIAMETER_LIGHT_SECONDS;
}



PopGeoLocation::PopGeoLocation(const GeoHelper* geo_helper,
							   const PopMultilateration* multilateration)
	: geo_helper_(geo_helper),
	  multilateration_(multilateration)
{
	assert(geo_helper != NULL);
	assert(multilateration != NULL);
}

bool PopGeoLocation::calculate_location(const vector<PopSighting>& sightings,
										double* lat, double* lng) const
{
	assert(lat != NULL);
	assert(lng != NULL);

	// Convert all the sightings from lat/long to (x,y,z) coordinates.
	vector<tuple<double, double, double, double> > sets(sightings.size());

	for (vector<PopSighting>::size_type i = 0; i < sightings.size(); ++i) {
		const PopSighting& sighting = sightings[i];

		// For now, assume that all base stations are at altitude 0.
		// TODO(snyderek): Should the base stations report their altitudes in
		// addition to lat/long?
		double x, y, z;
		tie(x, y, z) = geo_helper_->turn_llh_into_xyz(sighting.lat,
													  sighting.lng, 0.0,
													  "wgs84");

		const double t = spherical_distance_to_linear(sighting.frac_secs);

		sets[i] = make_tuple(x / GeoHelper::SPEED_OF_LIGHT_M_PER_S,
							 y / GeoHelper::SPEED_OF_LIGHT_M_PER_S,
							 z / GeoHelper::SPEED_OF_LIGHT_M_PER_S,
							 t);
	}

	// Do the multilateration.
	tuple<double, double, double> tracker_xyz;
	if (!multilateration_->calculate_xyz(sets, &tracker_xyz))
		return false;

	double tracker_x, tracker_y, tracker_z;
	tie(tracker_x, tracker_y, tracker_z) = tracker_xyz;

	double temp_lat, temp_lng, temp_alt;
	tie(temp_lat, temp_lng, temp_alt) = geo_helper_->turn_xyz_into_llh(
		tracker_x * GeoHelper::SPEED_OF_LIGHT_M_PER_S,
		tracker_y * GeoHelper::SPEED_OF_LIGHT_M_PER_S,
		tracker_z * GeoHelper::SPEED_OF_LIGHT_M_PER_S,
		"wgs84");

	*lat = temp_lat;
	*lng = temp_lng;

	return true;
}

}
