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
	assert(sightings.size() >= 3u);
	assert(lat != NULL);
	assert(lng != NULL);

	// Convert all the sightings from lat/long to (x,y,z) coordinates. For now,
	// only use the first three sightings in the computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	vector<tuple<double, double, double, double> > sets(3);

	for (vector<PopSighting>::size_type i = 0; i < 3u; ++i) {
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
	double temp_lat = 0.0, temp_lng = 0.0, temp_alt = 0.0;
	tie(temp_lat, temp_lng, temp_alt) = calculate_3input_location(sets);

	*lat = temp_lat;
	*lng = temp_lng - 180;
}

// Take 3 positions readings in the format:
// [(x, y, z, time)] * 3
// and return a location (x, y, z)
// Based on the finalreport_trilateration.pdf, section 8.10

tuple<double, double, double> PopMultilateration::calculate_3input_location(
	const vector<tuple<double, double, double, double> >& sets) const
{
	assert(sets.size() == 3u);

	// First establish coordinate system using point 1 (offset 0) as the
	// origin, and point 2 as the vector for the coordinate system such
	// that point 2 lies on the line y2=0, z2=0:

	double x1, y1, z1, t1;
	double x2, y2, z2, t2;
	double x3, y3, z3, t3;

	tie(x1, y1, z1, t1) = sets[0];
	tie(x2, y2, z2, t2) = sets[1];
	tie(x3, y3, z3, t3) = sets[2];

	printf("Input data:\n");
	printf("%f %f %f %f\n", x1, y1, z1, t1);
	printf("%f %f %f %f\n", x2, y2, z2, t2);
	printf("%f %f %f %f\n", x3, y3, z3, t3);

	double ox, oy, oz;
	tie(ox, oy, oz) = make_tuple(-x1, -y1, -z1); // save the origin...

	// Translate all coordinates to the origin of the first point:
	printf(">  %f %f %f\n", x1, y1, z1); // Sanity check that this is 0
	tie(x1, y1, z1) = translate_xyz(x1, y1, z1, ox, oy, oz); // Better turn out to be 0!
	tie(x2, y2, z2) = translate_xyz(x2, y2, z2, ox, oy, oz);
	tie(x3, y3, z3) = translate_xyz(x3, y3, z3, ox, oy, oz);

	printf("translated Locations:\n");
	printf("%f %f %f\n", x1, y1, z1);
	printf("%f %f %f\n", x2, y2, z2);
	printf("%f %f %f\n", x3, y3, z3);

	// Undo and verify...
	double px, py, pz;
	double lat, lon, alt;

	tie(px, py, pz) = make_tuple(x3, y3, z3);
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz); // Better turn out to be 0!
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Verify Result:  %f %f %f\n", lat, lon-180.0, alt);
	//


	// find the angle in the Z plane of the line between p1 and p2

	const double theta1 = atan2(y2-y1, x2-x1) * -1.0;

	// translate the coordinates in the Z plane to put p2 on a straight line...
	tie(x1, y1, z1) = rotate_xyz_around_z(x1, y1, z1, theta1); // This should still be 0
	printf(">  %f %f %f\n", x1, y1, z1); // sanity check that this is still 0
	tie(x2, y2, z2) = rotate_xyz_around_z(x2, y2, z2, theta1); // This should now be inline on Z
	// y2 should be 0, or very very very close to 0 now.
	tie(x3, y3, z3) = rotate_xyz_around_z(x3, y3, z3, theta1);


	// Undo and verify...
	tie(px, py, pz) = make_tuple(x3, y3, z3);
	tie(px, py, pz) = rotate_xyz_around_z(px, py, pz, -theta1); // This should still be 0
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz); // Better turn out to be 0!
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Verify Result:  %f %f %f\n", lat, lon-180.0, alt);
	//


	// TODO: the above transform, the below transform, and the translation could
	// surely all be done more efficiently all at once, but this way we can see
	// that we aren't making mistakes.


	// find the angle around y to rotate to get z2 to be 0...

	const double theta2 = atan2(z2-z1, x2-x1);
	printf("==1=  %f %f %f\n", x2, y2, z2);
	tie(x1, y1, z1) = rotate_xyz_around_y(x1, y1, z1, theta2); // This should still be 0
	tie(x2, y2, z2) = rotate_xyz_around_y(x2, y2, z2, theta2); // This should now be inline on Z
	printf("==2=  %f %f %f\n", x2, y2, z2); // z2 should be very close to 0 now
	tie(x3, y3, z3) = rotate_xyz_around_y(x3, y3, z3, theta2);



	// Undo and verify...
	tie(px, py, pz) = make_tuple(x3, y3, z3);
	tie(px, py, pz) = rotate_xyz_around_y(px, py, pz, -theta2); // This should still be 0
	tie(px, py, pz) = rotate_xyz_around_z(px, py, pz, -theta1); // This should still be 0
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz); // Better turn out to be 0!
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Verify Result:  %f %f %f\n", lat, lon-180.0, alt);
	//


	// find the angle around x to rotate to get z3 to be 0...

	const double theta3 = -1.0 * atan2(z3-z1, y3-y1);
	tie(x1, y1, z1) = rotate_xyz_around_x(x1, y1, z1, theta3); // This should still be 0
	tie(x2, y2, z2) = rotate_xyz_around_x(x2, y2, z2, theta3); // x2 and y2 should still be near 0
	tie(x3, y3, z3) = rotate_xyz_around_x(x3, y3, z3, theta3); // z3 should be near 0.
	printf("==3=  %f %f %f\n", x3, y3, z3); // z3 should be very close to 0 now

	// Undo and verify...
	tie(px, py, pz) = make_tuple(x3, y3, z3);
	tie(px, py, pz) = rotate_xyz_around_x(px, py, pz, -theta3); // This should still be 0
	tie(px, py, pz) = rotate_xyz_around_y(px, py, pz, -theta2); // This should still be 0
	tie(px, py, pz) = rotate_xyz_around_z(px, py, pz, -theta1); // This should still be 0
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz); // Better turn out to be 0!
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Verify Result:  %f %f %f\n", lat, lon-180.0, alt);
	//

	// Ok, everything should be transformed into a coordinate system which is now
	// 1. the original point 1 is the origin
	// 2. point 2 is on the X axis, with y=0, z=0
	// 3. point 3 is in the xy plane, with z = 0

	printf("Locations:\n");
	printf("%f %f %f\n", x1, y1, z1);
	printf("%f %f %f\n", x2, y2, z2);
	printf("%f %f %f\n", x3, y3, z3);


	// Carrying on...
	const double L3 = sqrt(sqr(x3) + sqr(y3)); // length of the antenna pair baselines
	const double R13 = from_time_to_distance(t3-t1);
	const double R12 = from_time_to_distance(t2-t1);

	printf("Distance difference r1-3:  %f km\n", R13/1000.0);
	printf("Distance difference r1-2:  %f km\n", R12/1000.0);

	// Equation 8.16:
	const double u = (R13*(x2/R12) - x3) / y3;

	// Equation 8.17:
	const double v = ( sqr(L3) - sqr(R13) + (R13*R12) * (1.0 - sqr(x2/R12)) ) / (2.0 * y3);

	// Equation 8.20:
	const double d = -1.0*(1.0 - sqr(x2/R12) + sqr(u));

	// Equation 8.21:
	const double ve = x2 * (1.0 - sqr(x2/R12)) - 2.0*u*v;

	// Equation 8.22:
	const double f = (sqr(R12) / 4.0) * sqr(1.0 - sqr(x2/R12)) - sqr(v);

	// Equation 8.24:
	// dx^2 + ex + f = 0

	const double subs=sqr(ve) - 4.0*d*f;
	printf("e:  %f\n", ve);
	printf("d:  %f\n", d);
	printf("f:  %f\n", f);
	printf("Sqrt of this:  %f\n", subs);

	const double Xa = (-ve + sqrt(sqr(ve)-4.0*d*f)) / (2.0*d);
	const double Xb = (-ve - sqrt(sqr(ve)-4.0*d*f)) / (2.0*d);

	printf("X results:  %f %f\n", Xa, Xb);

	// now translate it all back...

	tie(px, py, pz) = make_tuple(Xa, 0.0, 0.0);
	tie(px, py, pz) = rotate_xyz_around_x(px, py, pz, -theta3);
	tie(px, py, pz) = rotate_xyz_around_y(px, py, pz, -theta2);
	tie(px, py, pz) = rotate_xyz_around_z(px, py, pz, -theta1);
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz);
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Result:  %f %f %f\n", lat, lon-180.0, alt);

	tie(px, py, pz) = make_tuple(Xb, 0.0, 0.0);
	tie(px, py, pz) = rotate_xyz_around_x(px, py, pz, -theta3);
	tie(px, py, pz) = rotate_xyz_around_y(px, py, pz, -theta2);
	tie(px, py, pz) = rotate_xyz_around_z(px, py, pz, -theta1);
	tie(px, py, pz) = translate_xyz(px, py, pz, -ox, -oy, -oz);
	tie(lat, lon, alt) = geo_helper_.turn_xyz_into_llh(px, py, pz, "wgs84");
	printf("Result:  %f %f %f\n", lat, lon-180.0, alt);



	return make_tuple(lat, lon, alt);
}

}
