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

private:
	const GeoHelper geo_helper_;
};

// The following functions are exposed so that they can be called from tests.

// Given sets of base station coordinates (x, y, z, frac_secs), returns the
// (x, y, z) coordinates of the tracker.
//
// Prerequisite: sets.size() == 5
boost::tuple<double, double, double> calculate_xyz(
	const std::vector<boost::tuple<double, double, double, double> >& sets);

// This function is the same as calculate_xyz, except that all coordinates
// (including the time value) must be translated so that sets[0] is
// (0.0, 0.0, 0.0, 0.0). The returned coordinates will have to be translated
// back to the original coordinate space to get a useful value.
boost::tuple<double, double, double> calculate_xyz_from_origin(
	const std::vector<boost::tuple<double, double, double, double> >& sets);

}

#endif
