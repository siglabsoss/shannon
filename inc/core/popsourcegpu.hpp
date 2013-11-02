/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SOURCE_GPU_HPP_
#define __POP_SOURCE_GPU_HPP_

#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

#include <iostream>

#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <boost/math/common_factor.hpp>

#include "core/popobject.hpp"
#include "core/popsinkgpu.hpp"
#include "core/popexception.hpp"
#include "mdl/poptimestamp.hpp"

#include <cuda.h>
//#include "dsp/utils.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cassert>
//#include <cufft.h>
#include <cuda/helper_cuda.h>

using namespace boost::posix_time;

namespace pop
{

/* Guaranteed minimum number of buffers for both sink and source. Must
 * be an even number or memory corruption */
#define POPSOURCE_GPU_NUM_BUFFERS 20

/* This class uses a single mirror, but the memory is real unlike mmap'd memory of PopSource.
 * This value is how many buffers the gpu sees
 */
#define DOUBLE_POPSOURCE_GPU_NUM_BUFFERS (POPSOURCE_GPU_NUM_BUFFERS*2)

/**
 * Data Source Class
 * Class is mostly a template to process sinks of data.
 */
template <typename OUT_TYPE>
class PopSourceGpu : public PopObject
{
public:
    /**
     * Class constructor.
     * @param sizeBuf The default length of the output buffer in samples. If
     * set to zero then no output buffer is allocated.
     */
	PopSourceGpu(const char* name, size_t nInBuf) :
        PopObject(name), m_buf(name, nInBuf), m_timestamp_buf(name, nInBuf), debug_free_buffers(0), m_rgSink(0), m_reqBufSize(nInBuf)
    {
    }

    /**
     * Class deconstructor.
     */
    ~PopSourceGpu()
    {
    	m_buf.free_circular_buffer(m_buf.m_bytesAllocated);
    	m_timestamp_buf.free_circular_buffer(m_timestamp_buf.m_bytesAllocated);
    }

//    // Debug function that prints out the contents of the entire mirroerd buffer, assumes that datatype is char
//    // See it in action at commit 8ef51ebb782e54d168206320a25aabe58e1f0d0f
//    void debug_print()
//    {
//    	using namespace std;
//    	char h_data[m_buf.m_bytesAllocated];
//
//    	cudaMemcpy(h_data, m_buf.m_d_bufPtr, m_buf.m_bytesAllocated * sizeof(char), cudaMemcpyDeviceToHost);
//
//    	cout << "data: " << endl << " ";
//    	for( size_t i = 0; i < m_buf.m_bytesAllocated; i++)
//    	{
//    		if( h_data[i] == 0 )
//    			cout << "0";
//    		else
//    			cout << h_data[i];
//
//    	}
//
//    	cout << endl;
//    }

    /**
     * Processes new source data and calls any connected sinks.
     */
    void process()
    {
    	size_t num_new_pts, num_new_timestamp_pts;

    	// num new points is always the buffer size
    	num_new_pts = num_new_timestamp_pts = m_reqBufSize;

        typename std::vector<PopSinkGpu<OUT_TYPE>* >::iterator it;
        size_t uncopied_pts, timestamp_uncopied_pts;
        size_t req_samples_from_sink, timestamp_req_samples_from_sink;


        OUT_TYPE* data_head = m_buf.m_d_bufPtr + m_buf.source_idx();
        PopTimestamp* timestamp_data_head = m_timestamp_buf.m_d_bufPtr + m_timestamp_buf.source_idx();


        // copy data to upper mirror
		cudaMemcpy(data_head + m_buf.m_sizeBuf, data_head, num_new_pts * sizeof(OUT_TYPE), cudaMemcpyDeviceToDevice);

        // do it again for the timestamp buffer
		cudaMemcpy(timestamp_data_head + m_timestamp_buf.m_sizeBuf, timestamp_data_head, num_new_pts * sizeof(PopTimestamp), cudaMemcpyDeviceToDevice);

        // This is a hack to build a vector so that the for loop below can remain similar to how it looks in PopSource.hpp
		// in reality PopSourceGpu is limited to one sink
        std::vector<PopSinkGpu<OUT_TYPE>* > m_rgSinks;

        // only enter loop if a sink has been connected
        if( m_rgSink )
        	m_rgSinks.push_back(m_rgSink);

//        /* iterate through list of a single GPU sinks and determine how many times
//           to call it. */
        for( it = m_rgSinks.begin(); it != m_rgSinks.end(); it++ )
        {
            // get source buffer index and number of uncopied points
            size_t &sink_idx_into_buffer = (*it)->m_sourceBufIdx;
            req_samples_from_sink = (*it)->sink_size();
            uncopied_pts = ((m_buf.m_bufIdx+m_buf.m_sizeBuf) - sink_idx_into_buffer) % m_buf.m_sizeBuf + num_new_pts;

            // get source buffer index and number of uncopied points
            // 'sink_idx_into_buffer' is a reference to the current sinks 'sourceBufIdx'
            size_t &timestamp_sink_idx_into_buffer = (*it)->m_timestampSourceBufIdx;
            timestamp_req_samples_from_sink = (*it)->sink_size();
            timestamp_uncopied_pts = ((m_timestamp_buf.m_bufIdx+m_timestamp_buf.m_sizeBuf) - timestamp_sink_idx_into_buffer) % m_timestamp_buf.m_sizeBuf + num_new_pts;


            // If there's no specific length requested then send all available.
            if( 0 == req_samples_from_sink )
            {
            	printf(RED "Error: Gpu sink with 0 requested samples (assumption is violated) for object %s" RESETCOLOR "\n", (*it)->get_name());
            }
            // Otherwise send req_samples_from_sink samples at a time.
            else while ( uncopied_pts >= req_samples_from_sink )
            {

            	(*it)->unblock( m_buf.m_d_bufPtr + m_buf.sink_idx(sink_idx_into_buffer), req_samples_from_sink,
            					m_timestamp_buf.m_d_bufPtr + m_timestamp_buf.sink_idx(timestamp_sink_idx_into_buffer), timestamp_req_samples_from_sink );

            	// modify and wrap sink pointer
            	sink_idx_into_buffer += req_samples_from_sink;
            	sink_idx_into_buffer %= m_buf.m_sizeBuf;
            	uncopied_pts -= req_samples_from_sink;

            	// modify and wrap sink timestamp pointer
            	timestamp_sink_idx_into_buffer += timestamp_req_samples_from_sink;
            	timestamp_sink_idx_into_buffer %= m_timestamp_buf.m_sizeBuf;
            	timestamp_uncopied_pts -= timestamp_req_samples_from_sink;
            }

            // if this debug option is set, this prints how many free buffers are avaliable
            static int iii = 0;
            using namespace std;
            int free_buffers = POPSOURCE_GPU_NUM_BUFFERS - (*it)->queue_size();
            if( debug_free_buffers )
            {
            	// only report every N times to reduce spam
            	if( iii++ % 15 == 0 )
            		cout << get_name() << " free buffers: " << free_buffers << endl;
            }

            // check for overflow
            if( (*it)->queue_size() >= POPSOURCE_GPU_NUM_BUFFERS )
                throw PopException(msg_object_overflow, get_name(),
                    (*it)->get_name());
        }

        // advance buffer pointer
        m_buf.m_bufIdx += num_new_pts;
        m_buf.m_bufIdx %= m_buf.m_sizeBuf;

        // advance timestamp buffer pointer
        m_timestamp_buf.m_bufIdx += num_new_timestamp_pts;
        m_timestamp_buf.m_bufIdx %= m_timestamp_buf.m_sizeBuf;
    }



public:
    /**
     * Function to subscribe other sinks into this objects source.
     */
    void connect(PopSinkGpu<OUT_TYPE> &sink)
    {
        // automatically grow buffer if needed
        if( sink.sink_size() * DOUBLE_POPSOURCE_GPU_NUM_BUFFERS > m_buf.m_sizeBuf )
        {
        	m_buf.resize_buffer(sink.sink_size());
        	m_timestamp_buf.resize_buffer(sink.sink_size());
        }

        // set read indices
        sink.m_sourceBufIdx = m_buf.m_bufIdx;
        sink.m_timestampSourceBufIdx = m_timestamp_buf.m_bufIdx;

        if( m_rgSink != 0 )
        {
        	printf(YELLOW);
        	printf(msg_warning_replacing_gpu_sink, get_name());
        	printf(RESETCOLOR "\r\n");
        }

        // store sink
        m_rgSink = &sink;

        // store source into the sink
        sink.m_rgSource = this;
    }


    // Ben's mirrored GPU double buffer
    template <typename BUFFER_TYPE>
    class PopSourceBufferGpu : public PopObject
    {
    public:

    	PopSourceBufferGpu(const char* name, size_t source_size) : PopObject(name), m_bufIdx(0), m_d_bufPtr(0), m_sizeBuf(0), m_bytesAllocated(0), m_reqBufSize(source_size) {}

    	// the index that the source keeps starts at 0, and wraps at m_sizeBuf
    	// This function calculates the offset into the data array for pointer arithmetic
    	// The source starts at 1/4 of the buffer up to 1/2, and then loops from 0 to 1/4
    	size_t source_idx()
    	{
    		size_t val;
    		val = (m_sizeBuf/2 + m_bufIdx) % m_sizeBuf;
    		return val;
    	}

    	// the index that the sink keeps starts at 0, and wraps at m_sizeBuf
    	// This function calculates the offset into the data array for pointer arithmetic.  The parameter is the sinks's 0-based index
    	// The sink starts at 1/4 of the buffer up to 3/4 in a contiguous segment before modulousing
    	size_t sink_idx(size_t idx)
    	{
    		size_t val;
    		val = m_sizeBuf/2 + idx;
    		return val;
    	}


    	/**
    	 * Guarentee that buffer (in samples) is evenly divisible by 4 such that double buffering (buffer mirroring) will work
    	 * with integer math.
    	 */
    	size_t calc_buffer_size( size_t size, size_t datatype_size, size_t sink_chunk_size )
    	{
    		size_t lcm;

    		// one quarter of the entire memory space in samples
    		size_t quarter_in_samples = (size / datatype_size) / 4;

    		// one quarter of the memory space must be divisible by the sink's data size
    		lcm = boost::math::lcm<size_t>(sink_chunk_size, quarter_in_samples);

    		// we also need an lcm between the sink and sources sample sizes
    		// m_lastReqSize is the same as m_reqBufSize
    		// and it also must be divisible by the sources data chunk
    		lcm = boost::math::lcm<size_t>(lcm, m_reqBufSize);

    		// convert back to size (bytes)
    		lcm *= datatype_size * 4;

    		return lcm;
    	}



    	/**
    	 * Create a circular buffer at least memSize bytes long.
    	 */
    	void* create_circular_buffer(size_t& bytes_allocated, size_t datatype_size, size_t sink_chunk_size)
    	{
    		void* temp_buf;
    		void* actual_buf;
    		void* mirror_buf;
    		int ret;

    		printf(GREEN);
    		printf(msg_create_new_circ_buf, get_name());
    		printf(msg_create_new_circ_buf_dbg_1, datatype_size, bytes_allocated);
    		printf(RESETCOLOR "\r\n");

    		// calculate how many pages required.  Note bytes_allocated is a reference so this also affects m_bytesAllocated
    		bytes_allocated = calc_buffer_size( bytes_allocated, datatype_size, sink_chunk_size );

    		checkCudaErrors(cudaMalloc(&actual_buf, bytes_allocated));

    		// allocates all bytes in buffer to 0
    		checkCudaErrors(cudaMemset(actual_buf, 0, bytes_allocated));

    		return actual_buf;
    	}

    	void free_circular_buffer(size_t size)
    	{
    		uint8_t* mirror_buf;

    		if( m_d_bufPtr )
    		{
    			printf(RED "Freeing circular buffer for object %s" RESETCOLOR "\n", get_name());

    			checkCudaErrors(cudaFree(m_d_bufPtr));
    		}
    	}

    	/**
    	 * Resize buffer. Make sure to call this before get_buffer to make sure
    	 * there's enough space.
    	 * @param sizeBuf Total buffer size in number of samples.
    	 * @param sink_chunk_size Requested chunk size by the sink in number of samples
    	 */
    	void resize_buffer(size_t sizeBuf, size_t sink_chunk_size = 1)
    	{
//    		// TODO: resize circular buffer instead of destroy
    		free_circular_buffer(m_bytesAllocated);


    		m_bytesAllocated = sizeBuf * sizeof(BUFFER_TYPE) * DOUBLE_POPSOURCE_GPU_NUM_BUFFERS;

    		m_d_bufPtr = (BUFFER_TYPE*)create_circular_buffer( m_bytesAllocated, sizeof(BUFFER_TYPE), sink_chunk_size );

    		// calculate the size of the buffer compensating for the fact that m_bytesAllocated is double
    		m_sizeBuf = m_bytesAllocated / ( sizeof(BUFFER_TYPE) * 2 );
    	}

    	/// Current Out Buffer index
    	size_t m_bufIdx;

    	/// Out Buffer
    	BUFFER_TYPE* m_d_bufPtr;

    	/// Out Buffer size in number of samples
    	size_t m_sizeBuf;

    	/// Total amount of memory (in bytes) allocated for Out Buffer
    	size_t m_bytesAllocated;

    	/// The size of the buffer ( this the same as m_reqBufSize of the PopSourceGpu)
    	size_t m_reqBufSize;

    }; //PopSourceBufferGpu

public:



private:

    template <typename BUFFER_TYPE>
    BUFFER_TYPE* get_buffer(size_t sizeBuf, PopSourceBufferGpu<BUFFER_TYPE> &buf)
    {
    	// automatically grow buffer if needed
    	if( sizeBuf * POPSOURCE_GPU_NUM_BUFFERS > buf.m_sizeBuf ) // don't use double macro here
    		buf.resize_buffer(sizeBuf);

    	// only called if no size requested and no sinks are connected
    	if( 0 == buf.m_d_bufPtr )
    		throw PopException(msg_no_buffer_allocated, get_name());

    	return buf.m_d_bufPtr + buf.source_idx();
    }



public:

    /**
     * Get next buffer. This is an efficient way of writing directly into
     * the objects data structure to avoid having to make a copy of the data.
     */
    OUT_TYPE* get_buffer()
    {
    	return get_buffer(m_reqBufSize, m_buf);
    }

    PopTimestamp* get_timestamp_buffer()
    {
    	return get_buffer(m_reqBufSize, m_timestamp_buf);
    }

    // buffer for main data
    PopSourceBufferGpu<OUT_TYPE> m_buf;

    // buffer for timestamp data
    PopSourceBufferGpu<PopTimestamp> m_timestamp_buf;
public:
    bool debug_free_buffers;

protected:

    /// Attached Sink Class
    PopSinkGpu<OUT_TYPE>* m_rgSink;

    /// Out Buffer size in number of samples. Calling get_buffer always returns this many samples
    size_t m_reqBufSize;

};


} // namespace pop

#endif // __POP_SOURCE_HPP_
