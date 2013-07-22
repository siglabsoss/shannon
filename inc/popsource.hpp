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

#include <boost/thread.hpp>

#include <popobject.hpp>
#include <popsink.hpp>
#include <popexception.hpp>

#define POPSOURCE_NUM_BUFFERS 20 // minimum 2 buffers

namespace pop
{

/**
 * Data Source Class
 * Note this is a polymorphic class which requires Run-Time Type Information
 * (RTTI) to be enabled in the compiler. On gcc this should be on by default.
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
    PopSource(const char* name = "PopSource", size_t sizeBuf = 0) :
        PopObject(name), m_bufIdx(0), m_bufPtr(0),
        m_sizeBuf(sizeBuf), m_prefSize(sizeBuf)
    {
        if( sizeBuf )
            m_bufPtr = create_circular_buffer( sizeBuf * sizeof(OUT_TYPE) );
    }
    ~PopSource()
    {
        free_circular_buffer(m_bufPtr, m_memSize);
    }

    /**
     * Resize buffer. Make sure to call this before get_buffer to make sure
     * there's enough space.
     * @param size Total buffer size in number of samples.
     */
    void resize_buffer(size_t size)
    {
        // set the preferred buffer size for convenient functions
        m_prefSize = size;

        if( size > m_sizeBuf )
        {
            // TODO: resize circular buffer instead of destroy

            free_circular_buffer(m_bufPtr, m_memSize);

            m_sizeBuf = size;

            m_bufPtr = create_circular_buffer( m_sizeBuf * sizeof(OUT_TYPE) );
        }
    }

    /**
     * Processes new source data and calls any connected sinks.
     */
    void process(OUT_TYPE* data, size_t num_new_pts)
    {
        typename std::vector<PopSink<OUT_TYPE>* >::iterator it;
        size_t uncopied_pts;
        size_t req_samples_from_sink;

        // check to see how much data has been received.
        if( num_new_pts > m_sizeBuf )
            throw PopException(msg_too_much_data);

        if( num_new_pts == 0 )
            throw PopException(msg_passing_invalid_amount_of_samples);

        if( 0 != m_prefSize )
            if( m_prefSize != num_new_pts )
                throw PopException(msg_passing_invalid_amount_of_samples);

        /* Check to see if this is an external buffer (i.e. one that was not
        previously allocated with get_buffer() ). If it is external, copy it */
        if( data != (m_bufPtr + m_bufIdx) )
        {
            check_for_overflows(num_new_pts);

            // TODO: make this a SSE2 memcpy
            memcpy(m_bufPtr + m_bufIdx, data, num_new_pts * sizeof(OUT_TYPE));
        }

        /* iterate through list of sources and determine how many times
           to call them. */
        for( it = m_rgSources.begin(); it != m_rgSources.end(); it++ )
        {
            // get source buffer index and number of uncopied points
            size_t &sink_idx_into_buffer = (*it)->m_sourceBufIdx;
            req_samples_from_sink = (*it)->sink_size();
            uncopied_pts = (m_bufIdx - sink_idx_into_buffer) % m_sizeBuf + num_new_pts;

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
        }

        // advance buffer pointer
        m_bufIdx += num_new_pts;
        m_bufIdx %= m_sizeBuf;
    }

    /**
     * Call this function with new source data where the size is the default
     * buffer size for the class.
     */
    void process(OUT_TYPE* out)
    {
        process( out, m_prefSize );
    }

    /**
     * Call this function when new data has been written into the buffer.
     */
    void process()
    {
        process( m_bufPtr + m_bufIdx, m_prefSize );
    }

    /**
     * Get next buffer. This is an efficient way of writing directly into
     * the objects data structure to avoid having to make a copy of the data.
     */
    OUT_TYPE* get_buffer()
    {
        /* before we give out a pointer for the user to write to we want to
           make sure that we will not cause an overflow. However, we're
           checking to make sure the next m_prefSize bytes are free. The
           user may write less than this, but its recommended that they
           don't write more. Also, if they write less, they may get an
           overflow earlier than they would normally expect. A better method
           is for the user to specify how many bytes they want, but this
           takes away flexibility. */
        check_for_overflows(m_prefSize);

        return m_bufPtr + m_bufIdx;
    }

public:
    /**
     * Function to subscribe other sinks into this objects source.
     */
    void connect(PopSink<OUT_TYPE> &sink)
    {
        // determine if output buffer is big enough for sink
        if( (sink.m_reqBufSize * POPSOURCE_NUM_BUFFERS) > m_sizeBuf )
        {
            // TODO: resize circular buffer instead of destroy

            free_circular_buffer(m_bufPtr, m_memSize);

            m_sizeBuf = sink.m_reqBufSize * POPSOURCE_NUM_BUFFERS;

            m_bufPtr =create_circular_buffer( m_sizeBuf * sizeof(OUT_TYPE) );
        }

        // set read index
        sink.m_sourceBufIdx = m_bufIdx;

        // store sink
        m_rgSources.push_back(&sink);
    }

private:
    /**
     * Guaranteed to give the next page size multiple for a given buffer size.
     */
    size_t calc_page_size( size_t size )
    {
        // make sure we are at minimum one page
        size |= sysconf(_SC_PAGESIZE) - 1;

        // algorithm for determining required pages
        size |= size >> 1;
        size |= size >> 2;
        size |= size >> 4;
        size |= size >> 8;
        size |= size >> 16;
        size |= size >> 32;
        size += 1;

        // make sure that we didn't wrap around
        PopAssert( size );

        return size;
    }

    /**
     * Create a circular buffer at least memSize bytes long.
     */
    OUT_TYPE* create_circular_buffer(size_t memSize)
    {
        void* temp_buf;
        void* actual_buf;
        void* mirror_buf;
        int ret;

        m_memSize = calc_page_size( memSize );

        m_sizeBuf = m_memSize / sizeof(OUT_TYPE);

        printf(GREEN);
        printf(msg_create_new_circ_buf, get_name());
        printf(msg_requested_mem_size, memSize, m_memSize, m_sizeBuf);
        printf(RESETCOLOR "\r\n");

        if( m_memSize % sizeof(OUT_TYPE) )
            throw PopException( msg_page_div_into_data );

        /* Temporarily allocate thrice the memory to make sure we have a
           contiguous cirular buffer address space. This is a bit wasteful
           but it's currently the only way to ensure contiguous space. Remember
           if you use this method to make sure to unmap the proper amount of
           memory (i.e. m_memSize * 3) ! */
        temp_buf = mmap(0, m_memSize * 3, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS, -1, 0);

        // map actual buffer (second third of address space) into first third of memory map
        actual_buf = ((uint8_t*)temp_buf) + m_memSize;
        ret = remap_file_pages(actual_buf, m_memSize, 0, 0, 0);

        if( ret )
            throw PopException( "#1 remap_file_pages=%d, errno=%s", ret, strerror(errno) );

        // map mirror buffer (third third of address space) into first third of memory map
        mirror_buf = ((uint8_t*)temp_buf) + (2 * m_memSize);
        ret = remap_file_pages(mirror_buf, m_memSize, 0, 0, 0);

        if( ret )
            throw PopException( "#2 remap_file_pages=%d, errno=%s", ret, strerror(errno) );

        // shrink the memory map back down to it's necessary size
        // TODO: don't know if this works as expected. are my original higher remappings
        // maintained? Who knows! Would be nice to use 66% less memory. BTW, if you
        // put this line back in, make sure to change the munmap size!!
        //mremap(temp_buf, m_memSize * 3, m_memSize, MREMAP_FIXED, temp_buf);

        return (OUT_TYPE*)actual_buf;
    }

    void free_circular_buffer(OUT_TYPE* buf, size_t size)
    {
        uint8_t* mirror_buf;

        if( buf )
        {
            printf(RED "Freeing circular buffer for object %s", get_name());

            mirror_buf = ((uint8_t*)m_bufPtr) - m_memSize;
            munmap( (void*)mirror_buf, size * 3);
        }
    }

    void check_for_overflows(size_t num_new_pts)
    {
        typename std::vector<PopSink<OUT_TYPE>* >::iterator it;

        /* Iterate through all the connected sinks to make sure that the 
           number of new points requested will not overwrite any existing
           process. */
        for( it = m_rgSources.begin(); it != m_rgSources.end(); it++ )
        {
            const OUT_TYPE* cur_source_buf_ptr =
                                      (*it)->get_current_sink_buffer_pointer();
            // check for overflows
            if( cur_source_buf_ptr )
                if( (((m_bufPtr + m_bufIdx) - cur_source_buf_ptr) %
                    m_sizeBuf + num_new_pts) > m_sizeBuf )
                    throw PopException(msg_object_overflow, get_name(),
                                       (*it)->get_name());
        }
    }

    /// Current Out Buffer index
    size_t m_bufIdx;

    /// Out Buffer
    OUT_TYPE* m_bufPtr;

    /// Out Buffer size in number of samples
    size_t m_sizeBuf;

    /// Preferred window size in number of samples
    size_t m_prefSize;

    /// Total amount of memory (in bytes) allocated for Out Buffer
    size_t m_memSize;

    /// Attached Classes
    std::vector<PopSink<OUT_TYPE>* > m_rgSources;
};


} // namespace pop

#endif // __POP_SOURCE_HPP_
