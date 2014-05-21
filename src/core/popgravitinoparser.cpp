#include <stddef.h>

#include <iostream>
#include <string>
#include <stdint.h>

#include "core/popgravitinoparser.hpp"
#include "core/basestationfreq.h"
#include "b64/b64.h"
#include "core/util.h"



using namespace std;


uint32_t parseUint32_t(std::string in)
{
	uint32_t result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

uint64_t parseUint64_t(std::string in)
{
	uint64_t result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

namespace pop
{



PopGravitinoParser::PopGravitinoParser(unsigned notused) : PopJsonRPC(0)
{
}

// call this from main() after all functions are setup to test data demodulation
//FIXME: remove in final version
// void PopArtemisRPC::mock(void)
// {
// 	if( handler )
// 	{
// 		handler->process(values, ARRAY_LEN(values), 0, 0);
// 	}
// }

void PopGravitinoParser::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
	cout << str << endl;
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *params, *p0, *p1, *p2;

	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}


// 	if( method.compare("rx") == 0 )
// 	{
// 		p0 = find_json_token(arr, "params[0]");
// 		p1 = find_json_token(arr, "params[1]");
// 		p2 = find_json_token(arr, "params[2]");
// 		if( p0 && p0->type == JSON_TYPE_STRING && p1 && p1->type == JSON_TYPE_NUMBER && p2 && p2->type == JSON_TYPE_NUMBER )
// 		{
// 			cout << "got rx" << endl;
// 			cout << str << endl;
// 
// 			unsigned long offset;
// 			istringstream ( FROZEN_GET_STRING(p1) ) >> offset;
// 
// 			double clockCorrection;
// 
// 			istringstream ( FROZEN_GET_STRING(p2) ) >> clockCorrection;
// 
// 			packet_rx( FROZEN_GET_STRING(p0), (uint32_t)offset, clockCorrection );
// //			rcp_log(std::string(tok->ptr, tok->len));
// 			//			respond_int(0, methodId);
// 		}
// 	}
// 
// 	if( method.compare("raw") == 0 )
// 	{
// 		params = find_json_token(arr, "params");
// 
// 		int j;
// 		char buf[128];
// 		uint64_t values[params->num_desc];
// 		uint32_t modulusCorrection = 0; // corrects for modulus events in incoming signal
// 
// 		for(j=0;j<params->num_desc-1;j++)
// 		{
// 			snprintf(buf, 128, "params[%d]", j);
// 			values[j] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf))) + modulusCorrection;
// 
// 			if( values[j] < values[j-1] && j != 0)
// 			{
// 				modulusCorrection += ARTEMIS_CLOCK_SPEED_HZ;
// 
// 				// bump current sample as well
// 				values[j] += ARTEMIS_CLOCK_SPEED_HZ;
// 			}
// 
// //			printf("val = %u", values[j]);
// 		}
// 
// 		// last sample is different
// 		snprintf(buf, 128, "params[%d]", params->num_desc-1);
// 		values[params->num_desc-1] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf)));
// 
// 		if( handler )
// 		{
// 			handler->process(values, params->num_desc, 0, 0);
// 		}
// 	}
}



// code pulled from '/home/joel/uhd/host/lib/types/time_spec.cpp
// because that file was compiled with incorrect flags and get_system_time() returns garbage
namespace pt = boost::posix_time;
PopTimestamp get_microsec_system_time(void){
	pt::ptime time_now = pt::microsec_clock::universal_time();
	pt::time_duration time_dur = time_now - pt::from_time_t(0);
	return PopTimestamp(
			time_t(time_dur.total_seconds()),
			long(time_dur.fractional_seconds()),
			double(pt::time_duration::ticks_per_second())
	);
}




void PopGravitinoParser::send_rpc(const char *rpc_string, size_t length)
{
	// Leading null. Send this character as a precaution, in case the previous
	// RPC was not terminated properly. It's safe to do this because if Artemis
	// receives two null characters in a row, it will just ignore the empty RPC.
	this->tx.process("\0", 1);

	this->tx.process(rpc_string, length);

	// Trailing null
	this->tx.process("\0", 1);
}

}
