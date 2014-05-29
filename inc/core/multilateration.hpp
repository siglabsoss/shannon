#ifndef __MULTILATERATION__
#define __MULTILATERATION__

#include <vector>

#include <boost/tuple/tuple.hpp>

namespace pop
{

boost::tuple<double, double, double> calculate_xyz_position(
	const std::vector<boost::tuple<double, double, double, double> >& sets);

}

#endif
