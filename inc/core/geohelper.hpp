#ifndef __GEO_HELPER__
#define __GEO_HELPER__

#include <map>
#include <string>

#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>

namespace pop
{

class GeoHelper
{
public:
	// The speed of light in meters per second
	static const double SPEED_OF_LIGHT_M_PER_S;
	// Approximate radius of the Earth in meters
	static const double EARTH_RADIUS_M;

	GeoHelper();

	boost::tuple<double, double, double> turn_llh_into_xyz(
		double lat_dec, double long_dec, double height,
		const std::string& coord_system) const;
	boost::tuple<double, double, double> turn_xyz_into_llh(
		double x, double y, double z, const std::string& coord_system) const;

private:
	boost::tuple<double, double, double> get_abe_values(
		const std::string& coord_system) const;

	std::map<std::string, boost::tuple<double, double, double> > abe_values_;
	mutable boost::mutex abe_values_mtx_;
};

}

#endif
