#include <stdio.h>

#include <string>

#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"

using boost::tie;
using pop::GeoHelper;
using std::string;

int main()
{
	GeoHelper geo_helper;

	const double lat_dec = 37.506794;
	const double long_dec = -122.204533;
	const double height = 0.0;
	const string coord_system = "wgs84";

	double x = 0.0, y = 0.0, z = 0.0;
	tie(x, y, z) = geo_helper.turn_llh_into_xyz(lat_dec, long_dec, height,
												coord_system);

	// Should be x == -2699861.0, y == -4286555.0, z == 3862162.0.
	printf("x == %f, y == %f, z == %f\n", x, y, z);

	return 0;
}
