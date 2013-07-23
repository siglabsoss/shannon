/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SIGNAL_HPP_
#define __POP_SIGNAL_HPP_

#include <vector>
#include <iterator>
#include <cstring>
#include <complex>
#include <cstdio>

#include <boost/shared_ptr.hpp>

#include "core/popobject.hpp"
#include "core/popqueue.hpp"
#include "core/popmessage.hpp"
#include "core/popslot.hpp"

namespace pop
{

/**
 * Signal Class.
 * Emits signals to a slot.
 */
class PopSignal : public PopObject
{
public:
    /**
     * Class constructor.
     */
    PopSignal(const char* name = "PopSignal") : PopObject(name) { }

    /**
     * Class destructor.
     */
    ~PopSignal() { }

    /**
     * Connects a PopSlot to the current signal.
     */
    void connect(PopSlot &slot)
    {
        std::vector<PopSlot *>::iterator it;

        // do nothing if slot is already connected
        for( it = m_slots.begin(); it != m_slots.end(); it++ )
            if( *it == &slot ) return;

        m_slots.push_back(&slot);
    }

    /**
     * Disconnect a PopSlot from the current signal.
     */
    void disconnect(PopSlot &slot);

    void start()
    {
        PopMsgSharedPtr track( new PopMessage("MSG_TRACK") );

        //track->add("time",5.5);
        //track->add("address","0:0:0:124f:b3ee");

        push(track);
    }

protected:
    /**
     * Pushes the data to any attached signals
     */
    void push(PopMsgSharedPtr msg)
    {
        std::vector<PopSlot *>::iterator it;

        // do nothing if slot is already connected
        for( it = m_slots.begin(); it != m_slots.end(); it++ )
            (*it)->receive( this, msg );
    }

private:
    std::vector<PopSlot *> m_slots;
};

}

#endif // __POP_SIGNAL_HPP_
