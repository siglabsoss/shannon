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
#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using boost::get;
using boost::make_tuple;
using boost::tie;
using boost::tuple;
using std::vector;

namespace pop
{

const int PopMultilateration::MIN_NUM_BASESTATIONS = 5;

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

}  // namespace

PopMultilateration::PopMultilateration()
{
}

void PopMultilateration::calculate_location(
	const vector<PopSighting>& sightings, double* lat, double* lng) const
{
	assert(lat != NULL);
	assert(lng != NULL);

	// Convert all the sightings from lat/long to (x,y,z) coordinates. For now,
	// only use the first five sightings in the computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	vector<tuple<double, double, double, double> > sets(
		PopMultilateration::MIN_NUM_BASESTATIONS);
	assert(sightings.size() >= sets.size());

	for (vector<PopSighting>::size_type i = 0; i < sets.size(); ++i) {
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
	double tracker_x, tracker_y, tracker_z;
	tie(tracker_x, tracker_y, tracker_z) = calculate_xyz(sets);

	double temp_lat, temp_lng, temp_alt;
	tie(temp_lat, temp_lng, temp_alt) = geo_helper_.turn_xyz_into_llh(
		tracker_x, tracker_y, tracker_z, "wgs84");

	*lat = temp_lat;
	*lng = temp_lng;
}

tuple<double, double, double> calculate_xyz(
	const vector<tuple<double, double, double, double> >& sets)
{
	assert(!sets.empty());

	printf("\n================================================================="
	       "===============\n\nInput:\n");
	for (vector<tuple<double, double, double, double> >::const_iterator it =
			 sets.begin();
		 it != sets.end(); ++it) {
		const tuple<double, double, double, double>& tup = *it;
		printf("( %25.16f , %25.16f , %25.16f , %19.16f )\n",
			   get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
	}

	double ox, oy, oz, ot;
	tie(ox, oy, oz, ot) = sets[0];

	vector<tuple<double, double, double, double> > sets_transposed(sets.size());

	for (vector<tuple<double, double, double, double> >::size_type i = 0u;
		 i < sets.size(); ++i) {
		double x, y, z, t;
		tie(x, y, z, t) = sets[i];
		sets_transposed[i] = make_tuple(x - ox, y - oy, z - oz, t - ot);
	}

	// Do the multilateration.
	double tracker_x, tracker_y, tracker_z;
	tie(tracker_x, tracker_y, tracker_z) = calculate_xyz_from_origin(
		sets_transposed);

	const tuple<double, double, double> result =
		make_tuple(tracker_x + ox, tracker_y + oy, tracker_z + oz);

	printf("\nOutput:\n( %25.16f , %25.16f , %25.16f )\n\n====================="
	       "===========================================================\n\n",
		   get<0>(result), get<1>(result), get<2>(result));

	return result;
}

tuple<double, double, double> calculate_xyz_from_origin(
	const vector<tuple<double, double, double, double> >& sets)
{
	assert(static_cast<int>(sets.size()) ==
		   PopMultilateration::MIN_NUM_BASESTATIONS);

	printf("\nInput:\n");
	for (vector<tuple<double, double, double, double> >::const_iterator it =
			 sets.begin();
		 it != sets.end(); ++it) {
		const tuple<double, double, double, double>& tup = *it;
		printf("( %25.16f , %25.16f , %25.16f , %19.16f )\n",
			   get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
	}

	const double v = SPEED_OF_LIGHT_M_PER_S;

	vector<double> x(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> y(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> z(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> t(PopMultilateration::MIN_NUM_BASESTATIONS);

	for (int m = 0; m < PopMultilateration::MIN_NUM_BASESTATIONS; ++m) {
		const tuple<double, double, double, double>& tup = sets[m];
		x[m] = get<0>(tup);
		y[m] = get<1>(tup);
		z[m] = get<2>(tup);
		t[m] = get<3>(tup);
	}

	vector<double> A(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> B(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> C(PopMultilateration::MIN_NUM_BASESTATIONS);
	vector<double> D(PopMultilateration::MIN_NUM_BASESTATIONS);

	for (int m = 0; m < 2; ++m) {
		A[m] = nan("NaN");
		B[m] = nan("NaN");
		C[m] = nan("NaN");
		D[m] = nan("NaN");
	}
	for (int m = 2; m < PopMultilateration::MIN_NUM_BASESTATIONS; ++m) {
		A[m] = (2 * x[m]) / (v * t[m]) - (2 * x[1]) / (v * t[1]);
		B[m] = (2 * y[m]) / (v * t[m]) - (2 * y[1]) / (v * t[1]);
		C[m] = (2 * z[m]) / (v * t[m]) - (2 * z[1]) / (v * t[1]);
		D[m] = v * t[m] - v * t[1]
			   - (sqr(x[m]) + sqr(y[m]) + sqr(z[m])) / (v * t[m])
			   + (sqr(x[1]) + sqr(y[1]) + sqr(z[1])) / (v * t[1]);
	}

	printf("\n");
	for (int m = 2; m < PopMultilateration::MIN_NUM_BASESTATIONS; ++m) {
		printf("A[%d] == %20.16f , B[%d] == %20.16f , C[%d] == %20.16f , "
			   "D[%d] == %26.16f\n",
			   m, A[m], m, B[m], m, C[m], m, D[m]);
	}

	const double result_x =
		-(B[2] * (C[4]*D[3] - C[3]*D[4]) + C[2] * (B[3]*D[4] - B[4]*D[3]) +
				  (B[4]*C[3] - B[3]*C[4]) * D[2]) /
		(A[2] * (B[4]*C[3] - B[3]*C[4]) + B[2] * (A[3]*C[4] - A[4]*C[3]) +
				 (A[4]*B[3] - A[3]*B[4]) * C[2]);
	const double result_y =
		(A[2] * (C[4]*D[3] - C[3]*D[4]) + C[2] * (A[3]*D[4] - A[4]*D[3]) +
				 (A[4]*C[3] - A[3]*C[4]) * D[2]) /
		(A[2] * (B[4]*C[3] - B[3]*C[4]) + B[2] * (A[3]*C[4] - A[4]*C[3]) +
				 (A[4]*B[3] - A[3]*B[4]) * C[2]);
	const double result_z =
		-(A[2] * (B[4]*D[3] - B[3]*D[4]) + B[2] * (A[3]*D[4] - A[4]*D[3]) +
				  (A[4]*B[3] - A[3]*B[4]) * D[2]) /
		(A[2] * (B[4]*C[3] - B[3]*C[4]) + B[2] * (A[3]*C[4] - A[4]*C[3]) +
				 (A[4]*B[3] - A[3]*B[4]) * C[2]);

	const tuple<double, double, double> result =
		make_tuple(result_x, result_y, result_z);

	printf("\nOutput:\n( %25.16f , %25.16f , %25.16f )\n",
		   get<0>(result), get<1>(result), get<2>(result));

	return result;
}

}
