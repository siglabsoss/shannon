#include <iostream>
#include <sstream>
#include <string>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "core/popgravitinoparser.hpp"
#include "core/popsighting.hpp"
#include "core/popsightingstore.hpp"
#include "core/basestationfreq.h"
#include "core/utilities.hpp"
#include "core/popjsonrpc.hpp"

#include "b64/b64.h"
#include "core/util.h"



using namespace std;


namespace pop
{

PopGravitinoParser::PopGravitinoParser(unsigned notused, PopSightingStore* sighting_store)
	: PopSink<char>("PopJsonRPCSink", 0), tx("PopJsonRPCResponse"), headValid(false), command(0),
	  sighting_store_(sighting_store)
{
	assert(sighting_store != NULL);
}


void PopGravitinoParser::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str, unsigned stream_index)
{
	cout << str << endl;
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *params, *p0, *p1, *p2, *p3, *p4, *p5;

	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}


	if( method.compare("bx_rx") == 0 )
	{
		// basestation name, lat, lng, tracker id, full seconds, frac seconds
		p0 = find_json_token(arr, "params[0]");
		p1 = find_json_token(arr, "params[1]");
		p2 = find_json_token(arr, "params[2]");
		p3 = find_json_token(arr, "params[3]");
		p4 = find_json_token(arr, "params[4]");
		p5 = find_json_token(arr, "params[5]");

		if( p0 && p0->type == JSON_TYPE_STRING &&
			p1 && p1->type == JSON_TYPE_NUMBER &&
			p2 && p2->type == JSON_TYPE_NUMBER &&
			p3 && p3->type == JSON_TYPE_NUMBER &&
			p4 && p4->type == JSON_TYPE_NUMBER &&
			p5 && p5->type == JSON_TYPE_NUMBER )
		{
			PopSighting sighting;

			sighting.hostname = FROZEN_GET_STRING(p0);
			sighting.lat = parseNumber<double>(FROZEN_GET_STRING(p1));
			sighting.lng = parseNumber<double>(FROZEN_GET_STRING(p2));
			// TODO(snyderek): What is the data type of the tracker ID?
			sighting.tracker_id = parseNumber<uint64_t>(FROZEN_GET_STRING(p3));
			sighting.full_secs = parseNumber<time_t>(FROZEN_GET_STRING(p4));
			sighting.frac_secs = parseNumber<double>(FROZEN_GET_STRING(p5));

			sighting_store_->add_sighting(sighting);
		}
	}

	if( method.compare("grav_boot") == 0 && idTok != 0)
	{
		ostringstream os;
		os << "{\"result\":[], \"id\":" << parseNumber<uint32_t>(FROZEN_GET_STRING(idTok)) << "}";
		string message = os.str();

		cout << message << endl;
	}
}

unsigned PopGravitinoParser::get_stream_index(wrapped_net_header_t &header)
{
	unsigned i;
	for(i = 0; i < remotes.size(); i++)
	{
		if( header.source_port == remotes[i].source_port && header.source == remotes[i].source )
		{
			return i;
		}
	}

	remotes.push_back(header);
	streams.push_back(std::vector<char>());

	return remotes.size()-1;
}

void PopGravitinoParser::init() {}

void PopGravitinoParser::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	// first bytes always have source ip/port
	wrapped_net_header_t header;
	memcpy(&header, data, sizeof(wrapped_net_header_t) );

	unsigned index = get_stream_index(header);

	std::vector<char>* stream = &streams[index];







	const char* real_data = data + sizeof(wrapped_net_header_t);
	size_t real_size = size - sizeof(wrapped_net_header_t);


	unsigned i;
	for( i = 0; i < real_size; i++ )
	{
		char c = real_data[i];

		if( c == 0 )
		{
			parse(index);
			stream->erase(stream->begin(),stream->end());
		}
		else
		{
			stream->push_back(c);
		}
	}
}


void PopGravitinoParser::parse(unsigned index)
{
	std::vector<char>* stream = &streams[index];

	unsigned len = stream->size();
	if( len == 0 )
		return;

	std::string str(stream->begin(),stream->end());

	const char *json = str.c_str();

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *idTok = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		// skip printing this message for simple newline messages.  if one string matches, it returns 0 which we then multiply
		if( ( str.compare("\r\n\r\n") * str.compare("\r\n") * str.compare("\n") * str.compare("\r") ) != 0)
		{
			cout << "problem with json string (" <<  str << ")" << endl;
		}
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

	// verify message has "method" key
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}

	// verify message has "params" key
	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	// "id" key is optional.  It's absence means the message will not get a response
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
		idTok = 0;
	}

	execute(methodTok, paramsTok, idTok, arr, str, index);
}

void PopGravitinoParser::rcp_log(std::string log)
{
	cout << log << endl;
}

uint16_t PopGravitinoParser::rpc_get_autoinc(void)
{
	static uint16_t val = 1;
	return val++;
}


}
