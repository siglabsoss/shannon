/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_ENDIAN_H
#define __POP_ENDIAN_H

#include "core/popblock.hpp"

namespace pop
{
	class PopByteEndianSwapFloat : public PopBlock<float, float>
	{
	public:
		PopByteEndianSwapFloat() : PopBlock<float, float>(65536, 65536) { }
	private:
		void init() { }
		void process(float* in, float* out, size_t size)
		{
			size_t n;

			for( n = 0; n < size; n++ )
				((int32_t*)out)[n] = __builtin_bswap32(((int32_t*)in)[n]);
		}
	};

	class PopByteEndianSwapComplex : public PopBlock<std::complex<float>, std::complex<float> >
	{
	public:
		PopByteEndianSwapComplex() : PopBlock<std::complex<float>, std::complex<float> >(65536, 65536) { }
	private:
		void init() { }
		void process(std::complex<float>* in, std::complex<float>* out, size_t size)
		{
			size_t n;

			for( n = 0; n < size; n++ )
				((int64_t*)out)[n] = __builtin_bswap64(((int64_t*)in)[n]);
		}
	};
}

#endif // __POP_ENDIAN_H
