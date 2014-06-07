/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_GEO_LOCATION__
#define __POP_GEO_LOCATION__

#include <vector>

#include "core/popsighting.hpp"

namespace pop
{

class GeoHelper;
class PopMultilateration;

class PopGeoLocation
{
public:
	PopGeoLocation(const GeoHelper* geo_helper,
				   const PopMultilateration* multilateration);

	// If there is enough information to calculate the tracker's location, this
	// method sets *lat and *lng to the tracker's latitude and longitude,
	// respectively, and returns true. Otherwise, this method returns false and
	// leaves *lat and *lng unchanged.
	bool calculate_location(const std::vector<PopSighting>& sightings,
							double* lat, double* lng) const;

private:
	const GeoHelper* const geo_helper_;
	const PopMultilateration* const multilateration_;
};

}

#endif
