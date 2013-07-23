/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SINK_HPP_
#define __POP_SINK_HPP_

#include <boost/thread.hpp>

#include "core/popobject.hpp"
#include "core/popqueue.hpp"

namespace pop
{

// forward declaration of PopSource
template<typename> class PopSource;

template<typename T>
struct buffer_read_pointer
{
    const T* data;
    size_t len;
    buffer_read_pointer(const T* d, size_t l) : data(d), len(l) {}
    buffer_read_pointer() : data(0), len(0) {}
};



/**
 * Data Sink Class
 * Note this is a polymorphic class which requires Run-Time Type Information
 * (RTTI) to be enabled in the compiler. On gcc this should be on by default.
 * Class is mostly a template to process sources of data.
 */
template <typename IN_TYPE = std::complex<float> >
class PopSink : public PopObject,
    private PopQueue<buffer_read_pointer<IN_TYPE> >
{
protected:
    /**
     * Class constructor.
     * @param nInBuf Size of input buffer in number of samples. A value of
     * zero indicates that the class can accept any number of input samples.
     */
    PopSink(const char* name, size_t nInBuf = 0) : PopObject(name), m_reqBufSize(nInBuf),
        m_sourceBufIdx(0), m_pThread(0), m_curSourceBufPtr(0)
    {
    }

    /**
     * Class destructor.
     */
    ~PopSink()
    {
        delete m_pThread;
    }

    /**
     * Needs to be implemented by child class to handle incoming data.
     */
    virtual void process(const IN_TYPE* in, size_t size) = 0;

    /**
     * Needs to be implemented by child class to initialize anything
     * that needs to be called from the same thread.
     */
    virtual void init() = 0;

public:
    /**
     * Start Thread
     */
    void start_thread()
    {
        if( 0 == m_pThread )
            m_pThread = new boost::thread(boost::bind(&PopSink::run, this));
    }

    /**
     * Returns requested sample size for sink.
     */
    size_t sink_size() {
        return m_reqBufSize;
    }

private:
    /**
     * Thread loop.
     */
    void run()
    {
        buffer_read_pointer<IN_TYPE> buf;

        init();

        while(1)
        {
            wait_and_pop( buf );

            set_current_sink_buffer_pointer( buf.data );

            process( buf.data, buf.len );
        }
    }

    /**
     * Get current source buffer pointer.
     */
    const IN_TYPE* get_current_sink_buffer_pointer()
    {
        boost::mutex::scoped_lock lock(m_csbpMutex);
        return m_curSourceBufPtr;
    }

    /**
     * Set current source buffer pointer.
     */
    void set_current_sink_buffer_pointer(const IN_TYPE* ptr)
    {
        boost::mutex::scoped_lock lock(m_csbpMutex);
        m_curSourceBufPtr = ptr;
    }

    /**
     * Called by connecting block to unblock data.
     */
    void unblock(const IN_TYPE* in, size_t size)
    {
        // check to for a valid amount of input samples
        if( 0 != m_reqBufSize )
            if( size != m_reqBufSize )
                throw PopException( msg_passing_invalid_amount_of_samples );

        if( 0 == size )
            throw PopException( msg_passing_invalid_amount_of_samples );

        if( m_pThread )
            push( buffer_read_pointer<IN_TYPE>(in,size) );
        else
            process( in, size );
    }

    /**
     * Helper function when the amount of data received is apriori known.
     */
    int unblock(IN_TYPE* const buf)
    {
        return unblock( buf, m_reqBufSize );
    }

    /// In Buffer size in number of samples
    size_t m_reqBufSize;

    /// In Buffer index in respective PopSource
    size_t m_sourceBufIdx;

    /// thread
    boost::thread *m_pThread;

    /// Current source buffer pointer
    const IN_TYPE* m_curSourceBufPtr;

    /// Current source buffer pointer access mutex
    boost::mutex m_csbpMutex;

    // friend classes
    template <typename> friend class PopSource;
};

} // namespace pop

#endif // __POP_SINK_HPP_
