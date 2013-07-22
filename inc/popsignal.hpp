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
#include <cstring>

#include <boost/shared_ptr.hpp>

namespace pop
{


class PopMessage : public PopObject
{
public:
    PopMessage() { set_name( "PopMessage" ); }
    ~PopMessage() { }
};


/**
 * Slot Class.
 * Receives signals from signal class.
 */
class PopSlot : public PopObject
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

private:
    
};


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
    PopSignal() { }

    /**
     * Class destructor.
     */
    ~PopSignal() { }

    /**
     * Connects a PopSlot to the current signal.
     */
    connect(PopSlot &slot);

    /**
     * Disconnect a PopSlot from the current signal.
     */
    disconnect(PopSlot &slot);

private:
    /**
     * Pushes the data to any attached signals
     */
    push(boost::shared_ptr<PopMessage> msg);



private:
    std::vector<PopSlot *> m_slots;
};

}

#endif // __POP_SIGNAL_HPP_
