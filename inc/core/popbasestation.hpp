/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BASE_STATION__
#define __POP_BASE_STATION__

#include <string>

namespace pop
{

class PopBaseStation
{
public:
	explicit PopBaseStation(const std::string& hostname) : hostname_(hostname)
	{
	}

	const std::string& hostname() const { return hostname_; }

private:
	const std::string hostname_;
};

}

#endif
