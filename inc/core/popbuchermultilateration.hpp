/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BUCHER_MULTILATERATION__
#define __POP_BUCHER_MULTILATERATION__

#include "core/popmultilateration.hpp"

namespace pop
{

class GeoHelper;

// Ralph Bucher and D. Misra, “A Synthesizable VHDL Model of the Exact Solution
// for Three-dimensional Hyperbolic Positioning System,” VLSI Design, vol. 15,
// no. 2, pp. 507-520, 2002. doi:10.1080/1065514021000012129
//
// http://www.hindawi.com/journals/vlsi/2002/935925/cta/
// Accessed June 6, 2014.
class PopBucherMultilateration : public PopMultilateration
{
public:
	explicit PopBucherMultilateration(const GeoHelper* geo_helper);

	virtual bool calculate_xyz(
		const std::vector<boost::tuple<double, double, double, double> >& sets,
		boost::tuple<double, double, double>* result) const;

private:
	// Minimum number of base stations needed for multilateration.
	static const int MIN_NUM_BASESTATIONS;

	const GeoHelper* const geo_helper_;

	double distance_from_earth_surface(double x, double y, double z) const;
};

}

#endif
