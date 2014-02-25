#ifndef __POP_GRAV_PARSER__
#define __POP_GRAV_PARSER__

/******************************************************************************
 * Copyright 2013 PopWi Technology Group, Inc. (PTG)
 *
 * This file is proprietary and exclusively owned by PTG or its associates.
 * This document is protected by international and domestic patents where
 * applicable. All rights reserved.
 *
 ******************************************************************************/


#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "core/objectstash.hpp"

#define POP_GRAVITINO_SUPPORTED_TOKENS 50


namespace pop
{

class PopGravitinoParser : public PopSink<char>
{
public:
	bool headValid;
	std::vector<unsigned char> command;
	ObjectStash radios;
	PopSource<PopRadio> tx;


	PopGravitinoParser();
	void init();
	void process(const char* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void parse();
};

} // namespace pop

#endif
