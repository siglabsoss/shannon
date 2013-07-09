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

#include <popobject.hpp>

namespace pop
{

/**
 * Data Sink Class
 * Note this is a polymorphic class which requires Run-Time Type Information
 * (RTTI) to be enabled in the compiler. On gcc this should be on by default.
 * Class is mostly a template to process sources of data.
 */
template <class IN_TYPE = std::complex<float> >
class PopSink : public PopObject
{
public:
    /**
     * Class constructor.
     * @param nInBuf Size of input buffer in number of samples. A value of
     * zero indicates that the class can accept any number of input samples.
     */
    PopSink(size_t nInBuf = 0, int is_threaded = 1) : m_reqBufSize(nInBuf),
        m_sourceBufIdx(0), m_pThread(0), m_pBarrier(0)
    {
        set_name("PopSink");

        if( is_threaded )
        {
            m_pBarrier = new boost::barrier(2);
            m_pThread = new boost::thread(boost::bind(&PopSink::run, this));
        }
    }

    /**
     * Class destructor.
     */
    ~PopSink()
    {
        delete m_pBarrier;
        delete m_pThread;
    }

    /**
     * Needs to be implemented by child class to handle incoming data.
     */
    virtual void process(IN_TYPE* in, size_t size) = 0;

    /**
     * Thread loop.
     */
     void run()
     {
        while(1)
        {
            m_pBarrier->wait();

            process( m_pInBuf, m_bufSize );
        }
     }

    /**
     * Called by connecting block to unblock data.
     */
    void unblock(IN_TYPE* in, size_t size)
    {
        // check to for a valid amount of input samples
        if( 0 != m_reqBufSize )
            if( size != m_reqBufSize )
                throw PopException( msg_passing_invalid_amount_of_samples );

        if( 0 == size )
            throw PopException( msg_passing_invalid_amount_of_samples );

        if( m_pThread )
        {
            m_pInBuf = in;
            m_bufSize = size;
            m_pBarrier->wait();
        }
        else
            process( in, size );
    }

    /**
     * Helper function when the amount of data received is apriori known.
     */
    int unblock(IN_TYPE* buf)
    {
        return unblock(buf, m_reqBufSize);
    }

public: // TODO: this should be protected, but having some issues with the friend statement
    /// In Buffer size in number of samples
    size_t m_reqBufSize;

    /// In Buffer index in respective PopSource
    size_t m_sourceBufIdx;

    /// In Buffer pointer (active at time of thread barrier release)
    IN_TYPE *m_pInBuf;

    /// In Buffer pointer length (active at time of thread barrier release)
    size_t m_bufSize;

    /// thread
    boost::thread *m_pThread;

    /// thread barrier
    boost::barrier *m_pBarrier;

    template <class OUT_TYPE> friend class PopSource;
};

} // namespace pop

#endif // __POP_SINK_HPP_
