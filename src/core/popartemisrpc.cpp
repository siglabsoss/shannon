#include "core/popartemisrpc.hpp"
#include "b64/b64.h"

#include <iostream>
#include <string>


using namespace std;


uint32_t parseUint32_t(std::string in)
{
	uint32_t result;
	std::stringstream ss;
	ss << in;
	ss >> result;
	return result;
}


namespace pop
{



PopArtemisRPC::PopArtemisRPC(unsigned notused) : PopJsonRPC(0), handler(0)
{
}


void PopArtemisRPC::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
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


	if( method.compare("rx") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		p1 = find_json_token(arr, "params[1]");
		p2 = find_json_token(arr, "params[2]");
		if( p0 && p0->type == JSON_TYPE_STRING && p1 && p1->type == JSON_TYPE_NUMBER && p2 && p2->type == JSON_TYPE_NUMBER )
		{
			cout << "got rx" << endl;
			cout << str << endl;

			unsigned long offset;
			istringstream ( FROZEN_GET_STRING(p1) ) >> offset;

			double clockCorrection;

			istringstream ( FROZEN_GET_STRING(p2) ) >> clockCorrection;

			packet_rx( FROZEN_GET_STRING(p0), (uint32_t)offset, clockCorrection );
//			rcp_log(std::string(tok->ptr, tok->len));
			//			respond_int(0, methodId);
		}
	}

	if( method.compare("raw") == 0 )
	{
		params = find_json_token(arr, "params");

//			cout << "got raw" << endl;
//			cout << params->num_desc << endl;
		int j;
		char buf[128];

		uint32_t values[params->num_desc];

		for(j=0;j<params->num_desc;j++)
		{
			snprintf(buf, 128, "params[%d]", j);
//			cout << FROZEN_GET_STRING(find_json_token(arr, buf)) << endl;
			values[j] = parseUint32_t(FROZEN_GET_STRING(find_json_token(arr, buf)));

//			cout << values[j] << endl;
		}

		if( handler )
		{
			handler->process(values, params->num_desc, 0, 0);
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

//int b64_decode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );

void PopArtemisRPC::packet_rx(std::string b64_serial, uint32_t offset, double clock_correction)
{
	// the clock correction is 48million cycles of the internal clock divided by the number of cycles for the pps period
	// this can be viewed as the crystal's tolerance (if we assume pps is 100% accurate)
	if( clock_correction > 1.8 || clock_correction < .2 )
	{
		cout << "Artemis probably doesn't have pps, dropping packet\r\n" << endl;
	}

	uint32_t maxOffset = 480000000; // in units of 10x
	PopTimestamp now = get_microsec_system_time();

//	cout << "now: " << now << endl;

	PopTimestamp m_timestamp_offset;
	// round system time to nearest second
	m_timestamp_offset = PopTimestamp(round(now.get_real_secs()));

//	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;

	// calculate the fraction component of seconds
	double fraction = (double) offset / maxOffset;

	// use += operator for lossless addition
	m_timestamp_offset += PopTimestamp(fraction);

//	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;

	// build a "packet" which contains the serial and packet object
	// we don't use the sink/source built in timestamp stream because we don't have a timestamp for each character in the transmission, just the entire thing
	boost::tuple<char[20], PopTimestamp> packet;
	strncpy(get<0>(packet), b64_serial.c_str(), 20);
	get<1>(packet) = m_timestamp_offset;


	// send
	packets.process(&packet, 1);


//	cout << "in packet_rx: " << b64_serial << endl;
//	unsigned encodedCount = b64_serial.length();
//	char serialDecoded[encodedCount];
//	unsigned decodedCount;
//	b64_decode(b64_serial.c_str(), encodedCount, serialDecoded, &decodedCount);
//
//
//	cout << "Serial: ";
//	for(unsigned i = 0; i<decodedCount;i++)
//	{
//		printf("%02x", serialDecoded[i]);
//	}
//	cout << endl;

}

}
