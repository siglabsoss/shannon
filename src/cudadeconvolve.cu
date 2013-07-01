/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <complex>

__global__ void deconvolve(std::complex<float> *pn, std::complex<float> *data, std::complex<float> *product)
{
	
}

extern "C" void start_deconvolve(std::complex<float> *pn,
								 std::complex<float> *data,
								 std::complex<float> *product,
                                 int len)
{
	deconvolve<<<1, len>>>(pn, data, product);
}
