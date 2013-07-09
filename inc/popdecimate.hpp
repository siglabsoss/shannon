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
	template <class T = std::complex<float> > class PopDecimate
	{
	public:
		PopDecimate(unsigned rate = 8, size_t nBuf = 65536) :
			m_rate(rate), m_nBuf(nBuf)
		{

		}
		~PopDecimate() { }

		void set_rate(unsigned rate)
		{
			m_rate = rate;
		}

		size_t size()
		{
			return nBuf;
		}

		void sink(T *data);

	private:
		size_t m_nBuf;
		unsigned m_rate;
	};
}

#endif // __POP_DECIMATE_H
