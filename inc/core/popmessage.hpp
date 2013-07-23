/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_MESSAGE_HPP_
#define __POP_MESSAGE_HPP_

#include <complex>
#include <stdint.h>

#include <boost/shared_ptr.hpp>

#include "core/popobject.hpp"

namespace pop
{

class PopMessage : public PopObject
{
public:
    PopMessage(const char* name = "PopMessage") : PopObject(name) {  }
    ~PopMessage() { }

    /**
     * Add data.
     */
     void add(const char* name, float data);
     void add(const char* name, double data);
     void add(const char* name, uint32_t data);
     void add(const char* name, uint8_t data);
     void add(const char* name, std::complex<float> data);
     void add(const char* name, std::complex<double> data);
     void add(const char* name, void* data, size_t bytes);
     void add(const char* name, const char* data);

private:
    const char* m_sender;
};

typedef boost::shared_ptr<PopMessage> PopMsgSharedPtr;

} // namespace pop

#endif // __POP_MESSAGE_HPP_
