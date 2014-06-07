/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// IMPORTANT: In this source file, all distances are measured in light-seconds
// and all times are measured in seconds, unless otherwise noted. This makes the
// math easier, because the speed of light in these units is 1.

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>

#include <boost/tuple/tuple.hpp>

#include "core/popfangmultilateration.hpp"

using boost::get;
using boost::make_tuple;
using boost::tuple;
using std::vector;

namespace pop
{

const int PopFangMultilateration::MIN_NUM_BASESTATIONS = 3;

PopFangMultilateration::PopFangMultilateration(const GeoHelper* geo_helper)
	: geo_helper_(geo_helper)
{
	assert(geo_helper != NULL);
}

bool PopFangMultilateration::calculate_xyz(
	const vector<tuple<double, double, double, double> >& sets,
	tuple<double, double, double>* result) const
{
	assert(result != NULL);

	// For now, only use the first MIN_NUM_BASESTATIONS sightings in the
	// computation.
	// TODO(snyderek): Use any additional sightings to improve accuracy.
	if (static_cast<int>(sets.size()) < MIN_NUM_BASESTATIONS)
		return false;

	printf("\nInput:\n");
	for (vector<tuple<double, double, double, double> >::const_iterator it =
			 sets.begin();
		 it != sets.end(); ++it) {
		const tuple<double, double, double, double>& tup = *it;
		printf("( %19.16f , %19.16f , %19.16f , %19.16f )\n",
			   get<0>(tup), get<1>(tup), get<2>(tup), get<3>(tup));
	}

	// TODO(snyderek): Implement this.

	tuple<double, double, double> result_temp = make_tuple(0.0, 0.0, 0.0);

	printf("\nOutput:\n( %19.16f , %19.16f , %19.16f )\n",
		   get<0>(result_temp), get<1>(result_temp), get<2>(result_temp));

	*result = result_temp;
	return true;
}

}
