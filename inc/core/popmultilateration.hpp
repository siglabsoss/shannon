/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_MULTILATERATION__
#define __POP_MULTILATERATION__

#include <map>
#include <string>
#include <vector>

#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>

#include "core/popsighting.hpp"

namespace pop
{

// Multilateration function that calculates the global position of a tracker,
// given the times when the tracking signal was received by different base
// stations. The function is implemented as a class in case there's any one-time
// initialization we wish to perform.
class PopMultilateration
{
public:
	PopMultilateration();

	void calculate_location(const std::vector<PopSighting>& sightings,
							double* lat, double* lng) const;

private:
	boost::tuple<double, double, double> calculate_3input_location(
		const std::vector<boost::tuple<double, double, double, double> >& sets)
		const;

	boost::tuple<double, double, double> get_abe_values(
		const std::string& coord_system) const;

	boost::tuple<double, double, double> turn_llh_into_xyz(
		double lat_dec, double long_dec, double height,
		const std::string& coord_system) const;
	boost::tuple<double, double, double> turn_xyz_into_llh(
		double x, double y, double z, const std::string& coord_system) const;

	std::map<std::string, boost::tuple<double, double, double> > abe_values_;
	mutable boost::mutex abe_values_mtx_;
};

}

#endif
