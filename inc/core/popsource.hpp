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
#include "mdl/poptimestamp.hpp"

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
        PopObject(name), m_buf(name), m_timestamp_buf(name)
    {
    }

    /**
     * Class deconstructor.
     */
    ~PopSource()
    {
    	m_buf.free_circular_buffer(m_buf.m_bytesAllocated);
    	m_timestamp_buf.free_circular_buffer(m_timestamp_buf.m_bytesAllocated);
    }

    /**
     * Legacy overload for calling process with no timestamp data
     */
    void process(const OUT_TYPE* data, size_t num_new_pts)
    {
    	process(data, num_new_pts, NULL, 0);
    }

    /**
     * Processes new source data and calls any connected sinks.
     */
    void process(const OUT_TYPE* data, size_t num_new_pts, const PopTimestamp* timestamp_data, size_t num_new_timestamp_pts)
    {
//        ptime t1, t2;
//        time_duration td;
//        t1 = microsec_clock::local_time();

        typename std::vector<PopSink<OUT_TYPE>* >::iterator it;
        size_t uncopied_pts, timestamp_uncopied_pts;
        size_t req_samples_from_sink, timestamp_req_samples_from_sink;

        // if no data is passed then do nothing
        if( 0 == num_new_pts )
        	return;

        // make sure that timestamp buffer is always allocated
        // this is because there is too much math below which assumes that m_sizeBuf is non 0
        if( m_timestamp_buf.m_sizeBuf < (1 * sizeof(PopTimestamp) * POPSOURCE_NUM_BUFFERS) )
        	m_timestamp_buf.resize_buffer(1);

        // If the data is from an external array then copy data into buffer.
        m_buf.fill_data(data, num_new_pts);

        // do it again for the timestamp buffer
        m_timestamp_buf.fill_data(timestamp_data, num_new_timestamp_pts);

        // change the index values for each timestamp we just copied in
        correct_timestamp_indices(m_buf, m_timestamp_buf, num_new_timestamp_pts);


        /* iterate through list of sources and determine how many times
           to call them. */
        for( it = m_rgSources.begin(); it != m_rgSources.end(); it++ )
        {
            // get source buffer index and number of uncopied points
            size_t &sink_idx_into_buffer = (*it)->m_sourceBufIdx;
            req_samples_from_sink = (*it)->sink_size();
            uncopied_pts = ((m_buf.m_bufIdx+m_buf.m_sizeBuf) - sink_idx_into_buffer) % m_buf.m_sizeBuf + num_new_pts;

            // get source buffer index and number of uncopied points
            // 'sink_idx_into_buffer' is a reference to the current sinks 'sourceBufIdx'
            size_t &timestamp_sink_idx_into_buffer = (*it)->m_timestampSourceBufIdx;
            timestamp_uncopied_pts = ((m_timestamp_buf.m_bufIdx+m_timestamp_buf.m_sizeBuf) - timestamp_sink_idx_into_buffer) % m_timestamp_buf.m_sizeBuf + num_new_timestamp_pts;


            // If there's no specific length requested then send all available.
            if( 0 == req_samples_from_sink )
            {
                (*it)->unblock(m_buf.m_bufPtr + sink_idx_into_buffer, uncopied_pts, m_timestamp_buf.m_bufPtr + timestamp_sink_idx_into_buffer, timestamp_uncopied_pts);

                sink_idx_into_buffer += uncopied_pts;
                sink_idx_into_buffer %= m_buf.m_sizeBuf;

                timestamp_sink_idx_into_buffer += timestamp_uncopied_pts;
                timestamp_sink_idx_into_buffer %= m_timestamp_buf.m_sizeBuf;
            }
            // Otherwise send req_samples_from_sink samples at a time.
            else while ( uncopied_pts >= req_samples_from_sink )
            {


            	//                    std::cout << "there are " << timestamp_uncopied_pts << " uncopied stamps" << std::endl;
            	//                    std::cout << "asking for " << req_samples_from_sink << " samples at a time" << std::endl;

            	size_t start = (size_t)-1;
            	size_t end = 0;

            	// loop through all the timestamps that we have not sent to the source yet
            	for( size_t i = 0; i < timestamp_uncopied_pts; i++ )
            	{
            		size_t offset = m_timestamp_buf.m_bufPtr[i+timestamp_sink_idx_into_buffer].offset;
            		//                    	std::cout << "  looking at offset #" << offset << std::endl;

            		// if the timestamp offest applies to data we are about to call unblock() with, record start and end
            		if( offset >= sink_idx_into_buffer || offset <= (sink_idx_into_buffer+req_samples_from_sink) )
            		{
            			start = std::min(start,i);
            			end = std::max(end,i);
            		}
            	}

            	// at this point start/end point to the first and last time samples that are relevant
            	// (this is awk because if (end - start) = 0, but this actually means 1 sample)
            	size_t timestamp_samples = (end-start)+1;

            	//                    std::cout << "    start: " << start << " end: " << end << std::endl;


            	size_t zero_timestamps_are_valid = 0;


            	// if start and end were never set, the loop above never matched conditions
            	if( start == (size_t)-1 && end == 0 )
            	{
            		// copy in 0 timestamps
            		timestamp_samples = 0;

            		// in this special case, we want to call unblock with the 3rd param pointing at the most recent timestamp, and the 4th being a 0
            		// the tail of the circular buffer is pointing at uninitialized space, so 'zero_timestamps_are_valid' bumps us back one, to the most recent valid timestamp
            		zero_timestamps_are_valid = 1;
            	}

            	(*it)->unblock( m_buf.m_bufPtr + sink_idx_into_buffer, req_samples_from_sink, m_timestamp_buf.m_bufPtr + timestamp_sink_idx_into_buffer - zero_timestamps_are_valid, timestamp_samples );

            	// modify and wrap sink pointer
            	sink_idx_into_buffer += req_samples_from_sink;
            	sink_idx_into_buffer %= m_buf.m_sizeBuf;
            	uncopied_pts -= req_samples_from_sink;

            	// modify and wrap sink timestamp pointer
            	timestamp_sink_idx_into_buffer += timestamp_samples;
            	timestamp_sink_idx_into_buffer %= m_timestamp_buf.m_sizeBuf;
            	timestamp_uncopied_pts -= timestamp_samples;
            }

            // check for overflow
            if( (*it)->queue_size() >= POPSOURCE_NUM_BUFFERS )
                throw PopException(msg_object_overflow, get_name(),
                    (*it)->get_name());
        }

        // advance buffer pointer
        m_buf.m_bufIdx += num_new_pts;
        m_buf.m_bufIdx %= m_buf.m_sizeBuf;

        // advance timestamp buffer pointer
        m_timestamp_buf.m_bufIdx += num_new_timestamp_pts;
        m_timestamp_buf.m_bufIdx %= m_timestamp_buf.m_sizeBuf;

//        t2 = microsec_clock::local_time();
//        td = t2 - t1;
        //std::cout << get_name() << " process " << num_new_pts << " points in time: " << td.total_microseconds() << "us" << std::endl;
    }

    /**
     * Call this function with new source data where the size is the default
     * buffer size for the class.
     */
    void process(OUT_TYPE* out)
    {
        process( out, m_buf.m_lastReqSize );
    }

    /**
     * Call this function when new data has been written into the buffer.
     */
    void process(size_t num_new_pts)
    {
        process( m_buf.m_bufPtr + m_buf.m_bufIdx, num_new_pts );
    }

    /**
     * Call this function when new data has been written into the buffer.
     */
    void process()
    {
        process( m_buf.m_bufPtr + m_buf.m_bufIdx, m_buf.m_lastReqSize );
    }

    /**
     * Call this function when new data has been written into buffer, and new timestamps are avaiable as well
     */
    void process(size_t num_new_pts, const PopTimestamp* timestamp_data, size_t num_new_timestamp_pts)
    {
    	process(m_buf.m_bufPtr + m_buf.m_bufIdx, num_new_pts, timestamp_data, num_new_timestamp_pts);
    }

    /**
     * Get next buffer. This is an efficient way of writing directly into
     * the objects data structure to avoid having to make a copy of the data.
     */
    OUT_TYPE* get_buffer(size_t sizeBuf)
    {
        // remember allocated size for process() helper function
    	m_buf.m_lastReqSize = sizeBuf;

        // automatically grow buffer if needed
        if( sizeBuf * POPSOURCE_NUM_BUFFERS > m_buf.m_sizeBuf )
        	m_buf.resize_buffer(sizeBuf);

        // only called if no size requested and no sinks are connected
        if( 0 == m_buf.m_bufPtr )
            throw PopException(msg_no_buffer_allocated, get_name());

        return m_buf.m_bufPtr + m_buf.m_bufIdx;
    }





public:
    /**
     * Function to subscribe other sinks into this objects source.
     */
    void connect(PopSink<OUT_TYPE> &sink)
    {
        // automatically grow buffer if needed
        if( sink.sink_size() * POPSOURCE_NUM_BUFFERS >m_buf.m_sizeBuf )
        	m_buf.resize_buffer(sink.sink_size());

        // set read indices
        sink.m_sourceBufIdx = m_buf.m_bufIdx;
        sink.m_timestampSourceBufIdx = m_timestamp_buf.m_bufIdx;

        // store sink
        m_rgSources.push_back(&sink);
    }


    void debug_print_timestamp_buffer()
    {
    	using namespace std;
    	cout << "timestamp buffer:" << endl;
//    	cout << "  location: " << m_timestamp_buf.m_bufPtr << endl;
    	cout << "  size: " << m_timestamp_buf.m_sizeBuf << endl;
    	cout << "  index: " << m_timestamp_buf.m_bufIdx << endl;

    	cout << "  ";

    	for( size_t i = 0; i < m_timestamp_buf.m_bufIdx; i++ )
    	{
    		cout << m_timestamp_buf.m_bufPtr[i].offset << ",";
    	}

    	cout << endl;
    }

    // returns: Are all the offsets for the timestamps in this buffer in order (only checks the first N)?
    bool timestamp_offsets_in_order(size_t count)
    {
    	bool in_order = true;

    	size_t last_offset = (size_t)-1; // WARNING: this is a weird way to get the maximum value for size_t

    	for(size_t i = 0; i < count; i++)
    	{
    		// the sample we want is the tail minus i, minus one.  (minus one because the tail always points to the next to be written object)
    		size_t offset = m_timestamp_buf.m_bufPtr[m_timestamp_buf.m_bufIdx-i-1].offset;

    		// skip the first check
    		// we are running backwards through the last N samples
    		// therefor if this offset is greater than or equal to the previous, set flag and break
    		if(offset >= last_offset)
    		{
    			in_order = false;
    			break;
    		}

    		last_offset = offset;
    	}

    	return in_order;
    }



    // Joel's magical circular buffer
    template <typename BUFFER_TYPE>
    class PopSourceBuffer : public PopObject
    {
    public:

    	PopSourceBuffer(const char* name) : PopObject(name), m_bufIdx(0), m_bufPtr(0), m_sizeBuf(0), m_bytesAllocated(0), m_lastReqSize(0) {}

    	/**
    	 * This code used to be inside process(), extracted here for DRY
    	 * Called from within process(); this optionally copies data into the buffer if it is from an external array
    	 */
    	void fill_data(const BUFFER_TYPE* data, size_t& num_new_pts)
    	{
    		// If the data is from an external array then copy data into buffer.
    		if( data != (m_bufPtr + m_bufIdx) )
    		{
    			// Automatically grow the buffer if there is not enough space.
    			if( num_new_pts * POPSOURCE_NUM_BUFFERS > m_sizeBuf )
    				resize_buffer(num_new_pts);

    			// Copy data into buffer
    			// TODO: make this a SSE2 memcpy
    			memcpy(m_bufPtr + m_bufIdx, data, num_new_pts * sizeof(BUFFER_TYPE));
    		}
    		else
    		{
    			// check to see how much data has been received.
    			if( num_new_pts > m_lastReqSize )
    				throw PopException(msg_too_much_data);
    		}
    	}

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

    	void free_circular_buffer(size_t size)
    	{
    		uint8_t* mirror_buf;

    		if( m_bufPtr )
    		{
    			printf(RED "Freeing circular buffer for object %s" RESETCOLOR "\n", get_name());

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

    		free_circular_buffer(m_bytesAllocated);

    		m_bytesAllocated = sizeBuf * sizeof(BUFFER_TYPE) * POPSOURCE_NUM_BUFFERS;

    		m_bufPtr = (BUFFER_TYPE*)create_circular_buffer( m_bytesAllocated, sizeof(BUFFER_TYPE) );

    		m_sizeBuf = m_bytesAllocated / sizeof(BUFFER_TYPE);
    	}

    	/// Current Out Buffer index
    	size_t m_bufIdx;

    	/// Out Buffer
    	BUFFER_TYPE* m_bufPtr;

    	/// Out Buffer size in number of samples
    	size_t m_sizeBuf;

    	/// Total amount of memory (in bytes) allocated for Out Buffer
    	size_t m_bytesAllocated;

    	/// Last requested buffer size
    	size_t m_lastReqSize;

    }; //PopSourceBuffer

private:
    void correct_timestamp_indices(PopSource::PopSourceBuffer<OUT_TYPE> &buf, PopSource::PopSourceBuffer<PopTimestamp> &tbuf, size_t new_stamps)
    {
    	// nothing to do
    	if( buf.m_bufIdx == 0 )
    		return;


    	for(size_t i = 0; i < new_stamps; i++)
    	{
    		tbuf.m_bufPtr[i+tbuf.m_bufIdx].offset += buf.m_bufIdx;
    	}
    }

private:

    // buffer for main data
    PopSourceBuffer<OUT_TYPE> m_buf;

    // buffer for timestamp data
    PopSourceBuffer<PopTimestamp> m_timestamp_buf;

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
