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

#include <complex>
#include <cstring>
#include <boost/thread.hpp>
#include <boost/signals2.hpp>

namespace pop
{
	class PopGpu
	{
	public:
		PopGpu();
		~PopGpu();
		void import(std::complex<float>* data, std::size_t len);
		void init();

		boost::signals2::signal<void (float*, std::size_t)> sig;

	private:
		void run();
		void crunch();
		void *cu_in_buf; ///< GPU input buffer
		void *cu_out_buf; ///< GPU output buffer

		size_t numSamples();

		std::complex<float> *mp_buffer; ///< circular buffer
		float *mp_product; ///< circular buffer
		std::complex<float> *mp_demod_func; ///< PN code vector
		size_t m_buf_size;
		size_t m_buf_read_idx;
		size_t m_buf_write_idx;

		boost::barrier *mp_barrier; ///< GPU data ready semaphore
		boost::thread *mp_thread; ///< GPU class process I/O thread
	};
}

#endif // __POP_GPU_H
