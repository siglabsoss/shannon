/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_OBJECT_HPP_
#define __POP_OBJECT_HPP_

#include <cstring>
#include <cstdlib>
#include <complex>
#include <vector>

#include "core/popexception.hpp"
#include "core/popassert.h"
#include "core/language.h"

#define SAMPLES1K   (1UL << 10)
#define SAMPLES2K   (1UL << 11)
#define SAMPLES4K   (1UL << 12)
#define SAMPLES8K   (1UL << 13)
#define SAMPLES16K  (1UL << 14)
#define SAMPLES32K  (1UL << 15)
#define SAMPLES64K  (1UL << 16)
#define SAMPLES128K (1UL << 17)
#define SAMPLES256K (1UL << 18)
#define SAMPLES512K (1UL << 19)
#define SAMPLES1M   (1UL << 20)
#define SAMPLES2M   (1UL << 21)
#define SAMPLES4M   (1UL << 22)
#define SAMPLES8M   (1UL << 23)
#define SAMPLES16M  (1UL << 24)
#define SAMPLES32M  (1UL << 25)
#define SAMPLES64M  (1UL << 26)
#define SAMPLES128M (1UL << 27)
#define SAMPLES256M (1UL << 28)
#define SAMPLES512M (1UL << 29)
#define SAMPLES1G   (1UL << 30)

namespace pop
{

/**
 * Parent class to all other classes.
 */
class PopObject
{
public:
    PopObject(const char* name = "PopObject") : m_name(name) { }
    ~PopObject() { }
    void set_name(const char* name) {m_name = name;}
    const char* get_name() {return m_name;}
private:
    const char *m_name;
};

} // namespace pop

#endif // __POP_OBJECT_HPP_
