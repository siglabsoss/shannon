#ifndef __POP_TOKENIZER__
#define __POP_TOKENIZER__

/******************************************************************************
 * Copyright 2013 PopWi Technology Group, Inc. (PTG)
 *
 * This file is proprietary and exclusively owned by PTG or its associates.
 * This document is protected by international and domestic patents where
 * applicable. All rights reserved.
 *
 ******************************************************************************/

//#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "mdl/popsymbol.hpp"
//#include "popradio.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
//#include "net/popnetworktimestamp.hpp"
//#include "json/json.h"
//#include "core/utilities.hpp"


//using boost::asio::ip::udp;
//using namespace boost::asio;
using namespace std;



namespace pop
{

class PopTokenizer : public PopSink<PopSymbol>, public PopSource<PopSymbol>
{
public:

	PopTokenizer() : PopSink<PopSymbol>( "PopTokenizer" ), PopSource<PopSymbol>( "PopTokenizer" )
	{}

	void init()
	{

	}



	void process(const PopSymbol* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
	{

	}




};
} // namespace pop


#endif
