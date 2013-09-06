/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SOURCE_HPP_
#define __POP_SOURCE_HPP_

#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

#include <iostream>

#include <boost/thread.hpp>
#include <boost/timer.hpp>
#include <boost/math/common_factor.hpp>

#include "core/popobject.hpp"
#include "core/popsink.hpp"
#include "core/popexception.hpp"

using namespace boost::posix_time;

namespace pop
{

/* Guaranteed minimum number of buffers for both sink and source. May be much
   larger than this due to memory page alignment. */
#define POPSOURCE_NUM_BUFFERS 20

/**
 * Data Source Class
 * Class is mostly a template to process sinks of data.
 */
template <typename OUT_TYPE = std::complex<float> >
class PopSource : public PopObject
{
protected:
    /**
     * Class constructor.
     * @param sizeBuf The default length of the output buffer in samples. If
     * set to zero then no output buffer is allocated.
     */
    PopSource(const char* name = "PopSource") :
        PopObject(name), m_bufIdx(0), m_bufPtr(0), m_sizeBuf(0),
        m_bytesAllocated(0), m_lastReqSize(0)
    {
    }

    /**
     * Class deconstructor.
     */
    ~PopSource()
    {
        free_circular_buffer(m_bufPtr, m_bytesAllocated);
    }


    /**
     * Processes new source data and calls any connected sinks.
     */
    void process(const OUT_TYPE* data, size_t num_new_pts)
    {
        ptime t1, t2;
        time_duration td;
        t1 = microsec_clock::local_time();

        typename std::vector<PopSink<OUT_TYPE>* >::iterator it;
        size_t uncopied_pts;
        size_t req_samples_from_sink;

        // if no connected sinks then do nothing
        if( 0 == m_rgSources.size() )
            return;

        // if no data is passed then do nothing
        if( 0 == num_new_pts )
            return;

        // If the data is from an external array then copy data into buffer.
        if( data != (m_bufPtr + m_bufIdx) )
        {
            // Automatically grow the buffer if there is not enough space.
            if( num_new_pts * POPSOURCE_NUM_BUFFERS > m_sizeBuf )
                resize_buffer(num_new_pts);

            // Copy data into buffer
            // TODO: make this a SSE2 memcpy
            memcpy(m_bufPtr + m_bufIdx, data, num_new_pts * sizeof(OUT_TYPE));
        }
        else
        {
            // check to see how much data has been received.
            if( num_new_pts > m_lastReqSize )
                throw PopException(msg_too_much_data);
        }

        /* iterate through list of sources and determine how many times
           to call them. */
        for( it = m_rgSources.begin(); it != m_rgSources.end(); it++ )
        {
            // get source buffer index and number of uncopied points
            size_t &sink_idx_into_buffer = (*it)->m_sourceBufIdx;
            req_samples_from_sink = (*it)->sink_size();
            uncopied_pts = ((m_bufIdx+m_sizeBuf) - sink_idx_into_buffer) % m_sizeBuf + num_new_pts;

            // If there's no specific length requested then send all available.
            if( 0 == req_samples_from_sink )
            {
                (*it)->unblock(m_bufPtr + sink_idx_into_buffer, uncopied_pts);

                sink_idx_into_buffer += uncopied_pts;
                sink_idx_into_buffer %= m_sizeBuf;
            }
            // Otherwise send req_samples_from_sink samples at a time.
            else while ( uncopied_pts >= req_samples_from_sink )
                {
                    (*it)->unblock( m_bufPtr + sink_idx_into_buffer,
                        req_samples_from_sink );

                    sink_idx_into_buffer += req_samples_from_sink;
                    sink_idx_into_buffer %= m_sizeBuf;
                    uncopied_pts -= req_samples_from_sink;
                }

            // check for overflow
            if( (*it)->queue_size() >= POPSOURCE_NUM_BUFFERS )
                throw PopException(msg_object_overflow, get_name(),
                    (*it)->get_name());
        }

        // advance buffer pointer
        m_bufIdx += num_new_pts;
        m_bufIdx %= m_sizeBuf;

        t2 = microsec_clock::local_time();
        td = t2 - t1;

        //std::cout << get_name() << " process " << num_new_pts << " points in time: " << td.total_microseconds() << "us" << std::endl;
    }

    /**
     * Call this function with new source data where the size is the default
     * buffer size for the class.
     */
    void process(OUT_TYPE* out)
    {
        process( out, m_lastReqSize );
    }

    /**
     * Call this function when new data has been written into the buffer.
     */
    void process(size_t num_new_pts)
    {
        process( m_bufPtr + m_bufIdx, num_new_pts );
    }

    /**
     * Call this function when new data has been written into the buffer.
     */
    void process()
    {
        process( m_bufPtr + m_bufIdx, m_lastReqSize );
    }

    /**
     * Get next buffer. This is an efficient way of writing directly into
     * the objects data structure to avoid having to make a copy of the data.
     */
    OUT_TYPE* get_buffer(size_t sizeBuf)
    {
        // remember allocated size for process() helper function
        m_lastReqSize = sizeBuf;

        // automatically grow buffer if needed
        if( sizeBuf * POPSOURCE_NUM_BUFFERS > m_sizeBuf )
            resize_buffer(sizeBuf);

        // only called if no size requested and no sinks are connected
        if( 0 == m_bufPtr )
            throw PopException(msg_no_buffer_allocated, get_name());

        return m_bufPtr + m_bufIdx;
    }


public:
    /**
     * Function to subscribe other sinks into this objects source.
     */
    void connect(PopSink<OUT_TYPE> &sink)
    {
        // automatically grow buffer if needed
        if( sink.sink_size() * POPSOURCE_NUM_BUFFERS > m_sizeBuf )
            resize_buffer(sink.sink_size());

        // set read index
        sink.m_sourceBufIdx = m_bufIdx;

        // store sink
        m_rgSources.push_back(&sink);
    }


private:
    /**
     * Guaranteed to give the next page size multiple for a requested buffer
     * size. Buffer size will be expanded to fit both an integer number of page
     * tables and an integer number of data chunks. This can also be stated as
     * "The least common mulitple of PAGESIZE and CHUNK, greater than REQUEST"
     * JDB: tested 7/25/2013
     */
    size_t calc_page_size( size_t size, size_t chunk_size )
    {
        size_t lcm;

        // find the least common multiple greater than size
        lcm = boost::math::lcm<size_t>(sysconf(_SC_PAGESIZE), chunk_size);

        // ceil(size/lcm) * lcm
        size = ((size + lcm - 1) / lcm) * lcm;

        return size;
    }

    /**
     * Create a circular buffer at least memSize bytes long.
     */
    void* create_circular_buffer(size_t& bytes_allocated, size_t chunk_size)
    {
        void* temp_buf;
        void* actual_buf;
        void* mirror_buf;
        int ret;

        printf(GREEN);
        printf(msg_create_new_circ_buf, get_name());
        printf(msg_create_new_circ_buf_dbg_1, chunk_size, bytes_allocated);

        // calculate how many pages required
        bytes_allocated = calc_page_size( bytes_allocated, chunk_size );

        printf(msg_create_new_circ_buf_dbg_2, bytes_allocated);
        printf(RESETCOLOR "\r\n");

        /* Temporarily allocate thrice the memory to make sure we have a
           contiguous cirular buffer address space. This is a bit wasteful
           but it's currently the only way to ensure contiguous space. Remember
           if you use this method to make sure to unmap the proper amount of
           memory (i.e. bytes_allocated * 3) ! */
        temp_buf = mmap(0, bytes_allocated * 3, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS, -1, 0);

        // map actual buffer (second third of address space) into first third of memory map
        actual_buf = ((uint8_t*)temp_buf) + bytes_allocated;
        ret = remap_file_pages(actual_buf, bytes_allocated, 0, 0, 0);

        if( ret )
            throw PopException( "#1 remap_file_pages=%d, errno=%s", ret, strerror(errno) );

        // map mirror buffer (third third of address space) into first third of memory map
        mirror_buf = ((uint8_t*)temp_buf) + (2 * bytes_allocated);
        ret = remap_file_pages(mirror_buf, bytes_allocated, 0, 0, 0);

        if( ret )
            throw PopException( "#2 remap_file_pages=%d, errno=%s", ret, strerror(errno) );

        // shrink the memory map back down to it's necessary size
        // TODO: don't know if this works as expected. are my original higher remappings
        // maintained? Who knows! Would be nice to use 66% less memory. BTW, if you
        // put this line back in, make sure to change the munmap size!!
        //mremap(temp_buf, bytes_allocated * 3, bytes_allocated, MREMAP_FIXED, temp_buf);

        return actual_buf;
    }

    void free_circular_buffer(OUT_TYPE* buf, size_t size)
    {
        uint8_t* mirror_buf;

        if( buf )
        {
            printf(RED "Freeing circular buffer for object %s", get_name());

            mirror_buf = ((uint8_t*)m_bufPtr) - size;
            munmap( (void*)mirror_buf, size * 3);
        }
    }

    /**
     * Resize buffer. Make sure to call this before get_buffer to make sure
     * there's enough space.
     * @param sizeBuf Total buffer size in number of samples.
     */
    void resize_buffer(size_t sizeBuf)
    {
        // TODO: resize circular buffer instead of destroy

        free_circular_buffer(m_bufPtr, m_bytesAllocated);

        m_bytesAllocated = sizeBuf * sizeof(OUT_TYPE) * POPSOURCE_NUM_BUFFERS;

        m_bufPtr = (OUT_TYPE*)create_circular_buffer( m_bytesAllocated, sizeof(OUT_TYPE) );

        m_sizeBuf = m_bytesAllocated / sizeof(OUT_TYPE);
    }

    /// Current Out Buffer index
    size_t m_bufIdx;

    /// Out Buffer
    OUT_TYPE* m_bufPtr;

    /// Out Buffer size in number of samples
    size_t m_sizeBuf;

    /// Total amount of memory (in bytes) allocated for Out Buffer
    size_t m_bytesAllocated;

    /// Last requested buffer size
    size_t m_lastReqSize;

    // --------------------------------
    // JSON member variables
    // --------------------------------

    /// current JSON string
    std::ostringstream m_jsonString;

    /// Attached Classes
    std::vector<PopSink<OUT_TYPE>* > m_rgSources;

    // --------------------------------
    // JSON methods
    // --------------------------------
protected:
    void sendJSON();
    void commaAppender();
    void pushJSON(const char* key, float value);
    void pushJSON(const char* key, double value);
    void pushJSON(const char* key, uint8_t value);
    void pushJSON(const char* key, uint16_t value);
    void pushJSON(const char* key, uint32_t value);
    void pushJSON(const char* key, uint64_t value);
    void pushJSON(const char* key, int8_t value);
    void pushJSON(const char* key, int16_t value);
    void pushJSON(const char* key, int32_t value);
    void pushJSON(const char* key, int64_t value);
};


} // namespace pop

#endif // __POP_SOURCE_HPP_
