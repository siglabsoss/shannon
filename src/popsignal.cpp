/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <typeinfo>

#include "popsignal.hpp"


namespace pop
{
	PopSignal::push(boost::shared_ptr<void> data)
	{
		const char* name;

		name = typeid(&data)::name();
	}


}
