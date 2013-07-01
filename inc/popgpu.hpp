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
#include <boost/thread.hpp>

namespace pop
{
	class PopGpu
	{
	public:
		PopGpu();
		~PopGpu();
		void import(void* data, std::size_t len);
		void init();
	private:
		void run();
		void crunch();
		void *cu_in_buf; ///< GPU input buffer
		void *cu_out_buf; ///< GPU output buffer
		boost::barrier *mp_barrier; ///< GPU data ready semaphore
		boost::thread *mp_thread; ///< GPU class process I/O thread
	};
}

#endif // __POP_GPU_H
