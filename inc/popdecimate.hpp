/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_DECIMATE_H
#define __POP_DECIMATE_H

#include <complex>
#include <cstring>
#include <boost/thread.hpp>
#include <boost/signals2.hpp>

namespace pop
{
	class PopDecimate
	{
	public:
		PopDecimate(unsigned decimate_rate = 8);
		~PopDecimate();
		void import(float* data, std::size_t len);
		void set_rate(unsigned decimate_rate);

		boost::signals2::signal<void (float*, std::size_t)> sig;

	private:
		unsigned m_decimate_rate;
	};
}

#endif // __POP_DECIMATE_H
