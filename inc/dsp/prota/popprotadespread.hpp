/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_PROT_A_DESPREAD_H
#define __POP_PROT_A_DESPREAD_H

#include <complex>
#include <cstring>
#include <boost/thread.hpp>
#include <boost/signals2.hpp>
#include <boost/timer.hpp>

#include "core/popblock.hpp"

using namespace boost::posix_time;

namespace pop
{
	class PopProtADespread : public PopBlock<std::complex<float>, float>
	{
	public:
		PopProtADespread();
		~PopProtADespread();

	private:
		void process(const std::complex<float>* in, float* out, size_t len);
		void init();

		std::complex<float> *mp_demod_func; ///< PN code vector

		ptime tLastProcess; // temp

	};
}

#endif // __POP_PROT_A_DESPREAD_H
