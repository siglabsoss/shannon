/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_PROT_A_TDMA_BIN_H
#define __POP_PROT_A_TDMA_BIN_H

#include <complex>
#include <cstring>

#include "core/popsink.hpp"
//#include "core/popsourcemsg.hpp"

namespace pop
{
/**
 * PopWi Protocol A TDMA binning class.
 */
class PopProtATdmaBin : public PopSink<float>
{
public:
	PopProtATdmaBin() : PopSink<float>("PopProtATdmaBin", 65536) { }

private:
	void process(const std::complex<float>* in, size_t len)
	{
		// TODO: crunch data here
	}

	void init()
	{
		// TrODO: initialize here
	}

};

} // namespace pop

#endif // __POP_PROT_A_TDMA_BIN_H
