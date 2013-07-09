/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_CONTROL_H
#define __POP_CONTROL_H

#include "popassert.h"

namespace pop
{
	class PopControl
	{
	public:
		PopControl();
		~PopControl();

		POP_ERROR run();
	};
}

#endif // __POP_CONTROL_H
