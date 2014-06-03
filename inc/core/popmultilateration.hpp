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

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/geohelper.hpp"
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
	// Minimum number of base stations needed for multilateration.
	static const int MIN_NUM_BASESTATIONS;

	PopMultilateration();

	void calculate_location(const std::vector<PopSighting>& sightings,
							double* lat, double* lng) const;

	// The following method is public so that it can be called from tests.
	//
	// Given sets of base station coordinates (x, y, z, frac_secs), returns the
	// (x, y, z) coordinates of the tracker.  All distances are measured in
	// light-seconds.
	//
	// Prerequisite: sets.size() == MIN_NUM_BASESTATIONS
	boost::tuple<double, double, double> calculate_xyz(
		const std::vector<boost::tuple<double, double, double, double> >& sets)
		const;

private:
	double distance_from_earth_surface(double x, double y, double z) const;

	const GeoHelper geo_helper_;
};

}

#endif
