/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include "popdecimate.hpp"


namespace pop
{
	PopDecimate::PopDecimate(unsigned decimate_rate) : m_decimate_rate(decimate_rate)
	{
	}

	PopDecimate::~PopDecimate()
	{
	}

	void PopDecimate::import(float* data, std::size_t len)
	{
		float *out = (float*)malloc(sizeof(float)*len/8);
		size_t n;

		for( n = 0; n < len / 8; n++ )
		{
			out[n] = data[n * 8];
		}

		sig(out, len / 8);

		free(out);
	}

	void PopDecimate::set_rate(unsigned decimate_rate)
	{
		m_decimate_rate = decimate_rate;
	}

} // namespace pop
