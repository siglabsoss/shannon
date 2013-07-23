/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SLOT_HPP_
#define __POP_SLOT_HPP_

#include <vector>
#include <iterator>
#include <cstring>
#include <complex>
#include <cstdio>

#include <boost/shared_ptr.hpp>

#include "core/popobject.hpp"
#include "core/popqueue.hpp"
#include "core/popmessage.hpp"

namespace pop
{

/**
 * Slot Class.
 * Receives signals from signal class.
 */
class PopSlot : public PopObject, PopQueue<PopMsgSharedPtr>
{
public:
    /**
     * Class constructor.
     */
	PopSlot() { }

    /**
     * Class destructor.
     */
	~PopSlot() { }

    /**
     * Receives signal messages.
     */
    void receive(PopObject *sig, PopMsgSharedPtr msg)
    {
        push( msg );
        printf("received %s from %s\r\n", msg->get_name(), sig->get_name() );
    }

private:
    // message queue
    std::vector<PopMsgSharedPtr> m_msgQueue;

    /// thread
    boost::thread *m_pThread;
};

} // namespace pop

#endif // __POP_SLOT_HPP_
