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

#include "core/popsighting.hpp"

namespace pop
{

class PopMultilateration
{
public:
	PopMultilateration();

	void calculate_location(const std::vector<PopSighting>& sightings,
							double* lat, double* lng) const;

private:
};

}

#endif
