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

namespace pop
{

// Abstract class for a multilateration function that calculates the global
// position of a tracker, given the times when the tracking signal was received
// by different base stations.
class PopMultilateration
{
public:
	virtual ~PopMultilateration() {}

	// Given sets of base station coordinates (x, y, z, frac_secs), calculates
	// the (x, y, z) coordinates of the tracker. All distances are measured in
	// light-seconds.
	//
	// If there's not enough information to calculate the tracker's location,
	// this method returns false and leaves *result unchanged.
	virtual bool calculate_xyz(
		const std::vector<boost::tuple<double, double, double, double> >& sets,
		boost::tuple<double, double, double>* result) const = 0;
};

}

#endif
