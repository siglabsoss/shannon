/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_FANG_MULTILATERATION__
#define __POP_FANG_MULTILATERATION__

#include "core/popmultilateration.hpp"

namespace pop
{

class GeoHelper;

// Fang, Bertrand T. "Simple solutions for hyperbolic and related position
// fixes." Aerospace and Electronic Systems, IEEE Transactions on 26, no. 5
// (1990): 748-753.
//
// https://trac.v2.nl/export/7877/andres/Documentation/TDOA/Simple_Solutions_for_TDOA-fang.pdf
// Accessed June 6, 2014.
class PopFangMultilateration : public PopMultilateration
{
public:
	explicit PopFangMultilateration(const GeoHelper* geo_helper);

	virtual bool calculate_xyz(
		const std::vector<boost::tuple<double, double, double, double> >& sets,
		boost::tuple<double, double, double>* result) const;

private:
	// Minimum number of base stations needed for multilateration.
	static const int MIN_NUM_BASESTATIONS;

	const GeoHelper* const geo_helper_;
};

}

#endif
