/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_QUEUE_HPP_
#define __POP_QUEUE_HPP_

#include <queue>

#include <boost/thread.hpp>

namespace pop
{
template<typename T> class PopQueue
{
private:
    std::queue<T> the_queue;
    mutable boost::mutex the_mutex;
    boost::condition_variable the_condition_variable;
protected:
    void push(T const& data)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        the_queue.push(data);
        lock.unlock();
        the_condition_variable.notify_one();
    }

    bool empty() const
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.empty();
    }

    bool try_pop(T& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        if(the_queue.empty())
        {
            return false;
        }

        popped_value=the_queue.front();
        the_queue.pop();
        return true;
    }

    void wait_and_pop(T& popped_value)
    {
        boost::mutex::scoped_lock lock(the_mutex);
        while(the_queue.empty())
        {
            the_condition_variable.wait(lock);
        }

        popped_value=the_queue.front();
        the_queue.pop();
    }
public:
    size_t queue_size()
    {
        boost::mutex::scoped_lock lock(the_mutex);
        return the_queue.size();
    }

};

} // namespace pop

#endif // __POP_QUEUE_HPP_
