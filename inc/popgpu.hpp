/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_GPU_H
#define __POP_GPU_H


#include <cstring>

namespace pop
{
	class PopGpu
	{
	public:
		PopGpu();
		~PopGpu();
		void crunch(void* data, std::size_t len);
	};
}

#endif // __POP_GPU_H
