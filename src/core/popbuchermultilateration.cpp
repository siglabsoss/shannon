/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// IMPORTANT: In this source file, all distances are measured in light-seconds
// and all times are measured in seconds, unless otherwise noted. This makes the
// math easier, because the speed of light in these units is 1.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"
#include "core/popbuchermultilateration.hpp"

using boost::get;
using boost::make_tuple;
using boost::tie;
using boost::tuple;
using std::vector;

namespace pop
{

const int PopBucherMultilateration::MIN_NUM_BASESTATIONS = 4;

namespace
{

// The speed of light in meters per second
const double SPEED_OF_LIGHT_M_PER_S = 299792458.0;

}  // namespace

PopBucherMultilateration::PopBucherMultilateration(const GeoHelper* geo_helper)
	: geo_helper_(geo_helper)
{
	assert(geo_helper != NULL);
}

bool PopBucherMultilateration::calculate_xyz(
	const vector<tuple<double, double, double, double> >& sets,
	tuple<double, double, double>* result) const
{
	assert(result != NULL);

	// For now, only use the first MIN_NUM_BASESTATIONS sightings in the
	// computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	if (static_cast<int>(sets.size()) < MIN_NUM_BASESTATIONS)
		return false;

	printf("\nInput:\n");
	for (vector<tuple<double, double, double, double> >::const_iterator it =
			 sets.begin();
		 it != sets.end(); ++it) {
		const tuple<double, double, double, double>& tup = *it;
		printf("( %19.16f , %19.16f , %19.16f , %19.16f )\n",
			   get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
	}

	double ti=get<3>(sets[0]); double tk=get<3>(sets[2]); double tj=get<3>(sets[1]); double tl=get<3>(sets[3]);
	double xi=get<0>(sets[0]); double xk=get<0>(sets[2]); double xj=get<0>(sets[1]); double xl=get<0>(sets[3]);
	double yi=get<1>(sets[0]); double yk=get<1>(sets[2]); double yj=get<1>(sets[1]); double yl=get<1>(sets[3]);
	double zi=get<2>(sets[0]); double zk=get<2>(sets[2]); double zj=get<2>(sets[1]); double zl=get<2>(sets[3]);

	double xji=xj-xi; double xki=xk-xi; double xjk=xj-xk; double xlk=xl-xk;
	double xik=xi-xk; double yji=yj-yi; double yki=yk-yi; double yjk=yj-yk;
	double ylk=yl-yk; double yik=yi-yk; double zji=zj-zi; double zki=zk-zi;
	double zik=zi-zk; double zjk=zj-zk; double zlk=zl-zk;

	// Don't take the absolute value of these deltas. (The Bucher and Misra
	// paper is wrong.)
	double rij=ti-tj; double rik=ti-tk;
	double rkj=tk-tj; double rkl=tk-tl;

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

/*
	printf("\nxi == %.16f\nyi == %.16f\nzi == %.16f\nti == %.16f\n"
		   "xj == %.16f\nyj == %.16f\nzj == %.16f\ntj == %.16f\n"
		   "xk == %.16f\nyk == %.16f\nzk == %.16f\ntk == %.16f\n"
		   "xl == %.16f\nyl == %.16f\nzl == %.16f\ntl == %.16f\n",
		   xi, yi, zi, ti, xj, yj, zj, tj, xk, yk, zk, tk, xl, yl, zl, tl);

	printf("\nxji == %.16f\nxki == %.16f\nxjk == %.16f\nxlk == %.16f\n"
		   "xik == %.16f\nyji == %.16f\nyki == %.16f\nyjk == %.16f\n"
		   "ylk == %.16f\nyik == %.16f\nzji == %.16f\nzki == %.16f\n"
		   "zik == %.16f\nzjk == %.16f\nzlk == %.16f\n",
		   xji, xki, xjk, xlk, xik, yji, yki, yjk, ylk, yik, zji, zki, zik, zjk,
		   zlk);

	printf("\nrij == %.16f\nrik == %.16f\nrkj == %.16f\nrkl == %.16f\n",
		   rij, rik, rkj, rkl);

	printf("\na == %.16f\nb == %.16f\nc == %.16f\nd == %.16f\ne == %.16f\n"
		   "f == %.16f\ng == %.16f\nh == %.16f\ni == %.16f\nj == %.16f\n"
		   "k == %.16f\nl == %.16f\nm == %.16f\nn == %.16f\no == %.16f\n",
		   a, b, c, d, e, f, g, h, i, j, k, l, m, n, o);
*/

	printf("\nPossible Solutions:\n( %19.16f , %19.16f , %19.16f )\n"
		   "( %19.16f , %19.16f , %19.16f )\n",
		   x1, y1, z1, x2, y2, z2);

	// There are two possible solutions to the multilateration equations. Choose
	// the one that's closer to the Earth's surface.
	tuple<double, double, double> result_temp;
	if (distance_from_earth_surface(x1, y1, z1) <
		distance_from_earth_surface(x2, y2, z2)) {
		result_temp = make_tuple(x1, y1, z1);
	} else {
		result_temp = make_tuple(x2, y2, z2);
	}

	printf("\nOutput:\n( %19.16f , %19.16f , %19.16f )\n",
		   get<0>(result_temp), get<1>(result_temp), get<2>(result_temp));

	*result = result_temp;
	return true;
}

double PopBucherMultilateration::distance_from_earth_surface(double x, double y,
															 double z) const
{
	double lat, lng, alt;
	tie(lat, lng, alt) = geo_helper_->turn_xyz_into_llh(
		x * SPEED_OF_LIGHT_M_PER_S, y * SPEED_OF_LIGHT_M_PER_S,
		z * SPEED_OF_LIGHT_M_PER_S, "wgs84");

	printf("lat == %.16f, lng == %.16f, alt == %.16f\n", lat, lng, alt);

	return fabs(alt / SPEED_OF_LIGHT_M_PER_S);
}

}
