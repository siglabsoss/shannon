/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

//#include <cstdint>

#include <core/popsource.hpp>

using namespace std;

namespace pop
{

long ostringstream_length(std::ostringstream &s)
{
	// save current position
	long cur = s.tellp();

	// seek to end
	s.seekp(0, ios::end);

	// grab length
	long length = s.tellp();

	// seek back to where we were
	s.seekp(cur);

	// profit
	return length;
}

template <>
void PopSource<char>::sendJSON()
{
	ostringstream ss;


	ss << "{ " << m_jsonString.str() << " }";

	long jsonLength = ostringstream_length(ss);

	char header[3];

	size_t headerLength;

	PopSource::build_message_header(jsonLength, header, &headerLength);

	// send header
	process(header, headerLength);

	// send json message, followed by null
	process(ss.str().c_str(), jsonLength+1);

	// emtpy m_jsonString according to http://stackoverflow.com/questions/624260/how-to-reuse-an-ostringstream
	m_jsonString.clear();
//	m_jsonString.seekp(0); // for outputs: seek put ptr to start
	m_jsonString.str("");
}




template <>
void PopSource<char>::commaAppender()
{
	long length = ostringstream_length(m_jsonString);

	if( length > 0 )
		m_jsonString << ",";
}

template <>
void PopSource<char>::pushJSON(const char* key, float value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, double value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, uint8_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, uint16_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}
template <>

void PopSource<char>::pushJSON(const char* key, uint32_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, uint64_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, int8_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, int16_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, int32_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}

template <>
void PopSource<char>::pushJSON(const char* key, int64_t value)
{
	commaAppender();
	m_jsonString << " \"" << key << "\": " << value;
}


}
