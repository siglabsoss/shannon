/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BIN_H
#define __POP_BIN_H

#include <complex>
#include <cstring>
#include <boost/thread.hpp>
#include <boost/signals2.hpp>

namespace pop
{
	class PopBin
	{
	public:
		PopBin();
		~PopBin();
		void import(float* data, std::size_t len);
		void init();

		boost::signals2::signal<void (float*, std::size_t)> sig;

	private:
	};
}

#endif // __POP_BIN_H
