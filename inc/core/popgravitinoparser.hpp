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

//#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "core/phymessage.h"
#include "mdl/poppeak.hpp"
#include "mdl/popsymbol.hpp"
#include <frozen/frozen.h>
//#include "popradio.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
//#include "core/utilities.hpp"


using namespace boost;
using namespace std;

#define POP_GRAVITINO_SUPPORTED_TOKENS 50


namespace pop
{

long parseLong(const std::string &in)
{
	long result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

double parseDouble(const std::string &in)
{
	double result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}


class PopGravitinoParser : public PopSink<char>
{
public:
	bool headValid;
	std::vector<unsigned char> command;
	ObjectStash radios;

//	vector<boost::tuple<unsigned short, const PopPeak*> > symbols[MAX_BASE_STATIONS];
//	PopTimestamp start_time;
//	PopTimestamp newest_time;
//	size_t last_comb;

	PopGravitinoParser() : PopSink<char>( "PopGravitinoParser", 1 ), headValid(false)
	{
//		// FIXME: grab from real clock
//		newest_time = start_time = PopTimestamp(1383178239.0 + 3);
//
//		// resize the pipeline of each basestation
//		for( int i = 0; i < MAX_BASE_STATIONS; i++ )
//		{
//			symbols[i].resize(PIPELINE_BINS);
//		}
//
//		cout << "bin count " << BIN_COUNT << endl;
//		cout << "PHY_MSG " << sizeof(PHY_MSG) << endl;
//		cout << "PHY_MSG_HELO " << sizeof(PHY_MSG_HELO) << endl;

	}

	void init()
	{
	}



	void process(const char* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
//		cout << "in process with '" << data << "'" << endl;

		if( data_size != 1 ) {
			cout << "Error " << this->get_name() << " may only accept 1 character at a time" << endl;
			return;
		}

		char c = data[0];

		if( !headValid )
		{
			if( c == 0 )
				headValid = true;
		}
		else
		{

			if( c == 0 )
			{
				parse();
				command.erase(command.begin(),command.end());
			}
			else
			{
				command.push_back(c);
			}
		}
	}

	void parse()
	{
		unsigned len = command.size();
		if( len == 0 )
			return;

		std::string str(command.begin(),command.end());

		cout << str << endl;

		const char *json = str.c_str();

		struct json_token arr[POP_GRAVITINO_SUPPORTED_TOKENS];
		const struct json_token *tok, *tok2, *tok3;

		// Tokenize json string, fill in tokens array
		int returnValue = parse_json(json, strlen(json), arr, POP_GRAVITINO_SUPPORTED_TOKENS);

		if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
		{
			cout << "problem with json string" << endl;
			return;
		}

		if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
		{
			cout << "problem with json string (too many things for us to parse)" << endl;
			return;
		}



		long serial;
		double lat,lng;

		std::string method, serialString;

		tok = find_json_token(arr, "serial");
		if( !(tok && tok->type == JSON_TYPE_NUMBER) )
		{
			return;
		}
		else
		{
			serial = parseLong(std::string(tok->ptr, tok->len));
		}

		tok = find_json_token(arr, "lat");
		if( !(tok && tok->type == JSON_TYPE_NUMBER) )
		{
			return;
		}
		else
		{
			lat = parseDouble(std::string(tok->ptr, tok->len));
		}

		tok = find_json_token(arr, "lng");
		if( !(tok && tok->type == JSON_TYPE_NUMBER) )
		{
			return;
		}
		else
		{
			lng = parseDouble(std::string(tok->ptr, tok->len));
		}


		PopRadio *r = radios[serial];
		r->setLat(lat);
		r->setLng(lng);

		cout << "built object: " << r->seralize() << endl;



	}



};
} // namespace pop


#endif
