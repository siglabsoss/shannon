// Geographical helper functions for nmea_info.py and friends
//
// Helps with geographic functions, including:
//  Lat+Long+Height -> XYZ
//  XYZ -> Lat+Long+Height
//  Lat+Long -> other Lat+Long (Helmert Transform)
//  Lat+Long -> easting/northing (OS GB+IE Only)
//  easting/northing -> Lat+Long (OS GB+IE Only)
//  OS easting/northing -> OS 6 figure ref
//
// See http://gagravarr.org/code/ for updates and information
//
// GPL
//
// Nick Burch - v0.06 (30/05/2007)

// Translated to C++ by PopWi.

#include <assert.h>
#include <math.h>

#include <map>
#include <string>

#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"

using boost::get;
using boost::make_tuple;
using boost::mutex;
using boost::tie;
using boost::tuple;
using std::map;
using std::string;

namespace pop
{

const double GeoHelper::SPEED_OF_LIGHT_M_PER_S = 299792458.0;
const double GeoHelper::EARTH_RADIUS_M = 6371000.0;

GeoHelper::GeoHelper()
{
	// For each co-ordinate system we do, what are the A, B and E2 values?
	// List is A, B, E^2 (E^2 calculated after)
	abe_values_["wgs84"] = make_tuple(6378137.0, 6356752.3141, -1.0);
	abe_values_["osgb"] = make_tuple(6377563.396, 6356256.91, -1.0);
	abe_values_["osie"] = make_tuple(6377340.189, 6356034.447, -1.0);

	for (map<string, tuple<double, double, double> >::iterator it =
			 abe_values_.begin();
		 it != abe_values_.end(); ++it) {
		const double a = get<0>(it->second);
		const double b = get<1>(it->second);
		const double e2 = (a*a - b*b) / (a*a);
		get<2>(it->second) = e2;
	}
}

// ##############################################################
// #             Generic Transform Functions                    #
// ##############################################################

tuple<double, double, double> GeoHelper::turn_llh_into_xyz(
	double lat_dec, double long_dec, double height,
	const string& coord_system) const
{
	// Convert Lat, Long and Height into 3D Cartesian x,y,z
	// See http://www.ordnancesurvey.co.uk/gps/docs/convertingcoordinates3D.pdf

	double a, b, e2;
	tie(a, b, e2) = get_abe_values(coord_system);

	const double theta = lat_dec  / 360.0 * 2.0 * M_PI;
	const double landa = long_dec / 360.0 * 2.0 * M_PI;

	const double v = a / sqrt( 1.0 - e2 * (sin(theta) * sin(theta)) );
	const double x = (v + height) * cos(theta) * cos(landa);
	const double y = (v + height) * cos(theta) * sin(landa);
	const double z = ( (1.0 - e2) * v + height ) * sin(theta);

	return make_tuple(x,y,z);
}

tuple<double, double, double> GeoHelper::turn_xyz_into_llh(
	double x, double y, double z, const string& coord_system) const
{
	// Convert 3D Cartesian x,y,z into Lat, Long and Height
	// See http://www.ordnancesurvey.co.uk/gps/docs/convertingcoordinates3D.pdf

	double a, b, e2;
	tie(a, b, e2) = get_abe_values(coord_system);

	const double p = sqrt(x*x + y*y);

	double lng = atan2(y, x);
	const double lat_init = atan( z / (p * (1.0 - e2)) );
	const double v = a / sqrt( 1.0 - e2 * (sin(lat_init) * sin(lat_init)) );
	double lat = atan( (z + e2*v*sin(lat_init)) / p );

	const double height = (p / cos(lat)) - v; // Ignore if a bit out

	// Turn from radians back into degrees
	lng = lng / 2.0 / M_PI * 360.0;
	lat = lat / 2.0 / M_PI * 360.0;

	return make_tuple(lat,lng,height);
}

tuple<double, double, double> GeoHelper::get_abe_values(
	const string& coord_system) const
{
	mutex::scoped_lock lock(abe_values_mtx_);

	const map<string, tuple<double, double, double> >::const_iterator it =
		abe_values_.find(coord_system);
	assert(it != abe_values_.end());
	return it->second;
}

}
