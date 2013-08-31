/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <cstdint>

#include <core/popsource.hpp>

using namepspace std;

namespace pop
{


void PopSource<char>::sendJSON()
{
	ostringstream ss;

	ss << "{ " << m_jsonString << " }"
	process(m_jsonString.str(), m_jsonString.size()+1);
}

void PopSource<char>::pushJSON(const char* key, float value)
{
	m_jsonString << " \"" << key << "\": " + value + ",";
}

void PopSource<char>::pushJSON(const char* key, double value)
{
}

void PopSource<char>::pushJSON(const char* key, uint8_t value)
{
}

void PopSource<char>::pushJSON(const char* key, uint16_t value)
{
}

void PopSource<char>::pushJSON(const char* key, uint32_t value)
{
}

void PopSource<char>::pushJSON(const char* key, uint64_t value)
{
}

void PopSource<char>::pushJSON(const char* key, int8_t value)
{
}

void PopSource<char>::sendJSON()
{

}
void PopSource<char>::pushJSON(const char* key, int16_t value)
{
}

void PopSource<char>::pushJSON(const char* key, int32_t value)
{
}

void PopSource<char>::pushJSON(const char* key, int64_t value)
{
}


}
