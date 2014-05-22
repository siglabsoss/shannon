/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <stddef.h>

#include <vector>

#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"

using std::vector;

namespace pop
{

PopMultilateration::PopMultilateration()
{
}

void PopMultilateration::calculate_location(
	const vector<PopSighting>& sightings, double* lat, double* lng) const
{
	assert(sightings.size() >= 3u);
	assert(lat != NULL);
	assert(lng != NULL);

	// TODO(snyderek): Perform the multilateration calculation.

	*lat = 0.0;
	*lng = 0.0;
}

}
