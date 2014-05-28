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
	PopMultilateration();

	void calculate_location(const std::vector<PopSighting>& sightings,
							double* lat, double* lng) const;

private:
	boost::tuple<double, double, double> calculate_3input_location(
		const std::vector<boost::tuple<double, double, double, double> >& sets)
		const;
	boost::tuple<double, double, double> foo(
		double ox, double oy, double oz,
		double x1, double y1, double z1, double t1,
		double x2, double y2, double z2, double t2,
		double x3, double y3, double z3, double t3) const;

	const GeoHelper geo_helper_;
};

}

#endif
