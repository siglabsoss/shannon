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
#include <boost/signals2/mutex.hpp>

#include "core/popobject.hpp"
#include "core/popqueue.hpp"
#include "mdl/poptimestamp.hpp"

namespace pop
{

// forward declaration of PopSource
template<typename> class PopSource;

template<typename T>
struct buffer_read_pointer
{
    const T* data;
    size_t len;

    const PopTimestamp* timestamp_data;
    size_t timestamp_len;

    buffer_read_pointer(const T* d, size_t l, const PopTimestamp* td, size_t tl) : data(d), len(l), timestamp_data(td), timestamp_len(tl) {}
    buffer_read_pointer() : data(0), len(0), timestamp_data(0), timestamp_len(0) {}
};



/**
 * Data Sink Class
 * Note this is a polymorphic class which requires Run-Time Type Information
 * (RTTI) to be enabled in the compiler. On gcc this should be on by default.
 * Class is mostly a template to process sources of data.
 */
template <typename IN_TYPE = std::complex<float> >
class PopSink : public PopObject,
    public PopQueue<buffer_read_pointer<IN_TYPE> >
{
protected:

    PopSink(const char* name, size_t nInBuf = 0);
    ~PopSink();

    /**
     * Needs to be implemented by child class to handle incoming data.
     */
    virtual void process(const IN_TYPE* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size) = 0;

    /**
     * Needs to be implemented by child class to initialize anything
     * that needs to be called from the same thread.
     */
    virtual void init() = 0;

public:

    void start_thread();
    size_t sink_size();

private:
    void run();
    void unblock(const IN_TYPE* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size );

    /// In Buffer size in number of samples
    size_t m_reqBufSize;

    /// In Buffer index in respective PopSource
    size_t m_sourceBufIdx;

    /// In timestamp Buffer index in respective PopSource
    size_t m_timestampSourceBufIdx;

    /// thread
    boost::thread *m_pThread;

    // initialize mutex
    boost::mutex m_mutex;

    // friend classes
    template <typename> friend class PopSource;

protected:
    // pointer to the source that feeds into us
    PopSource<IN_TYPE>* m_rgSource;
};

} // namespace pop

#endif // __POP_SINK_HPP_
