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
#include "b64/b64.h"
#include "core/util.h"



using namespace std;

namespace
{

template<typename T>
T parseNumber(const string& in)
{
	T result;
	stringstream ss;
	ss << in;
	ss >> result;
	return result;
}

}

namespace pop
{

PopGravitinoParser::PopGravitinoParser(unsigned notused,
									   PopSightingStore* sighting_store)
	: PopJsonRPC(0),
	  sighting_store_(sighting_store)
{
	assert(sighting_store != NULL);
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

}
