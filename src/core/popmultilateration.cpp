/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// IMPORTANT: In this source file, all distances are measured in light-seconds
// unless otherwise noted.

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

const int PopMultilateration::MIN_NUM_BASESTATIONS = 4;

namespace
{

// The speed of light in meters per second
const double SPEED_OF_LIGHT_M_PER_S = 299792458.0;
// Approximate radius of the Earth in meters
const double EARTH_RADIUS_M = 6371000.0;

inline double sqr(double x)
{
	return x * x;
}

// Given a distance measured along a great circle between two points on the
// Earth's surface, returns the straight-line distance between the two points in
// Euclidean space. This function assumes that the Earth is spherical.
double spherical_distance_to_linear(double dist_light_seconds)
{
	static const double EARTH_DIAMETER_LIGHT_SECONDS =
		EARTH_RADIUS_M * 2.0 / SPEED_OF_LIGHT_M_PER_S;

	return sin(dist_light_seconds / EARTH_DIAMETER_LIGHT_SECONDS) *
		EARTH_DIAMETER_LIGHT_SECONDS;
}

// This function is the same as calculate_xyz, except that all coordinates
// (including the time value) must be translated so that sets[0] is
// (0.0, 0.0, 0.0, 0.0). The returned coordinates will have to be translated
// back to the original coordinate space to get a useful value.
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
		printf("( %19.16f , %19.16f , %19.16f , %19.16f )\n",
			   get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
	}

	// Ralph Bucher and D. Misra, “A Synthesizable VHDL Model of the Exact
	// Solution for Three-dimensional Hyperbolic Positioning System,” VLSI
	// Design, vol. 15, no. 2, pp. 507-520, 2002.
	// doi:10.1080/1065514021000012129

	double ti=get<3>(sets[0]); double tk=get<3>(sets[2]); double tj=get<3>(sets[1]); double tl=get<3>(sets[3]);
	double xi=get<0>(sets[0]); double xk=get<0>(sets[2]); double xj=get<0>(sets[1]); double xl=get<0>(sets[3]);
	double yi=get<1>(sets[0]); double yk=get<1>(sets[2]); double yj=get<1>(sets[1]); double yl=get<1>(sets[3]);
	double zi=get<2>(sets[0]); double zk=get<2>(sets[2]); double zj=get<2>(sets[1]); double zl=get<2>(sets[3]);

	double xji=xj-xi; double xki=xk-xi; double xjk=xj-xk; double xlk=xl-xk;
	double xik=xi-xk; double yji=yj-yi; double yki=yk-yi; double yjk=yj-yk;
	double ylk=yl-yk; double yik=yi-yk; double zji=zj-zi; double zki=zk-zi;
	double zik=zi-zk; double zjk=zj-zk; double zlk=zl-zk;

	double rij=fabs(ti-tj); double rik=fabs(ti-tk);
	double rkj=fabs(tk-tj); double rkl=fabs(tk-tl);

	double s9 =rik*xji-rij*xki; double s10=rij*yki-rik*yji; double s11=rik*zji-rij*zki;
	double s12=(rik*(rij*rij + xi*xi - xj*xj + yi*yi - yj*yj + zi*zi - zj*zj)
	           -rij*(rik*rik + xi*xi - xk*xk + yi*yi - yk*yk + zi*zi - zk*zk))/2;

	double s13=rkl*xjk-rkj*xlk; double s14=rkj*ylk-rkl*yjk; double s15=rkl*zjk-rkj*zlk;
	double s16=(rkl*(rkj*rkj + xk*xk - xj*xj + yk*yk - yj*yj + zk*zk - zj*zj)
	           -rkj*(rkl*rkl + xk*xk - xl*xl + yk*yk - yl*yl + zk*zk - zl*zl))/2;

	double a= s9/s10; double b=s11/s10; double c=s12/s10; double d=s13/s14;
	double e=s15/s14; double f=s16/s14; double g=(e-b)/(a-d); double h=(f-c)/(a-d);
	double i=(a*g)+b; double j=(a*h)+c;
	double k=rik*rik+xi*xi-xk*xk+yi*yi-yk*yk+zi*zi-zk*zk+2*h*xki+2*j*yki;
	double l=2*(g*xki+i*yki+zki);
	double m=4*rik*rik*(g*g+i*i+1)-l*l;
	double n=8*rik*rik*(g*(xi-h)+i*(yi-j)+zi)+2*l*k;
	double o=4*rik*rik*((xi-h)*(xi-h)+(yi-j)*(yi-j)+zi*zi)-k*k;
	double s28=n/(2*m);     double s29=(o/m);       double s30=(s28*s28)-s29;
	double root=sqrt(s30);
	double z1=s28+root;
	double z2=s28-root;
	double x1=g*z1+h;
	double x2=g*z2+h;
	double y1=a*x1+b*z1+c;
	double y2=a*x2+b*z2+c;

	const tuple<double, double, double> result = make_tuple(x2, y2, z2);

	printf("\nOutput:\n( %19.16f , %19.16f , %19.16f )\n",
		   get<0>(result), get<1>(result), get<2>(result));

	return result;
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
	// only use the first MIN_NUM_BASESTATIONS sightings in the computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	vector<tuple<double, double, double, double> > sets(MIN_NUM_BASESTATIONS);
	assert(sightings.size() >= sets.size());

	for (vector<PopSighting>::size_type i = 0; i < sets.size(); ++i) {
		const PopSighting& sighting = sightings[i];

		// For now, assume that all base stations are at altitude 0.
		// TODO(snyderek): Should the base stations report their altitudes in
		// addition to lat/long?
		double x, y, z;
		tie(x, y, z) = geo_helper_.turn_llh_into_xyz(sighting.lat, sighting.lng,
													 0.0, "wgs84");

		const double t = spherical_distance_to_linear(sighting.frac_secs);

		sets[i] = make_tuple(x / SPEED_OF_LIGHT_M_PER_S,
							 y / SPEED_OF_LIGHT_M_PER_S,
							 z / SPEED_OF_LIGHT_M_PER_S,
							 t);
	}

	// Do the multilateration.
	double tracker_x, tracker_y, tracker_z;
	tie(tracker_x, tracker_y, tracker_z) = calculate_xyz(sets);

	double temp_lat, temp_lng, temp_alt;
	tie(temp_lat, temp_lng, temp_alt) = geo_helper_.turn_xyz_into_llh(
		tracker_x * SPEED_OF_LIGHT_M_PER_S,
		tracker_y * SPEED_OF_LIGHT_M_PER_S,
		tracker_z * SPEED_OF_LIGHT_M_PER_S,
		"wgs84");

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
		printf("( %19.16f , %19.16f , %19.16f , %19.16f )\n",
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

	printf("\nOutput:\n( %19.16f , %19.16f , %19.16f )\n\n====================="
	       "===========================================================\n\n",
		   get<0>(result), get<1>(result), get<2>(result));

	return result;
}

}
