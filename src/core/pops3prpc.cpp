#include <iostream>
#include <string>
#include <stdint.h>

#include "core/pops3prpc.hpp"
#include "core/basestationfreq.h"
#include "b64/b64.h"
#include "core/util.h"
#include "core/utilities.hpp"



using namespace std;


namespace pop
{



PopS3pRPC::PopS3pRPC(unsigned notused) : PopJsonRPC(0)//, handler(0)
{
}

void PopS3pRPC::execute_result(const struct json_token *resultTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{

}

void PopS3pRPC::execute_rpc(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
	cout << "got  " << str << endl;
//	std::string method = FROZEN_GET_STRING(methodTok);
//	const struct json_token *params, *p0, *p1, *p2;
//
//	if( method.compare("log") == 0 )
//	{
//		p0 = find_json_token(arr, "params[0]");
//		if( p0 && p0->type == JSON_TYPE_STRING )
//		{
//			rcp_log(FROZEN_GET_STRING(p0));
////			respond_int(0, methodId);
//		}
//	}
//
//
//	if( method.compare("rx") == 0 )
//	{
//		p0 = find_json_token(arr, "params[0]");
//		p1 = find_json_token(arr, "params[1]");
//		p2 = find_json_token(arr, "params[2]");
//		if( p0 && p0->type == JSON_TYPE_STRING && p1 && p1->type == JSON_TYPE_NUMBER && p2 && p2->type == JSON_TYPE_NUMBER )
//		{
//			cout << "got rx" << endl;
//			cout << str << endl;
//
//			unsigned long offset;
//			istringstream ( FROZEN_GET_STRING(p1) ) >> offset;
//
//			double clockCorrection;
//
//			istringstream ( FROZEN_GET_STRING(p2) ) >> clockCorrection;
//
//			packet_rx( FROZEN_GET_STRING(p0), (uint32_t)offset, clockCorrection );
////			rcp_log(std::string(tok->ptr, tok->len));
//			//			respond_int(0, methodId);
//		}
//	}
//
//	if( method.compare("raw") == 0 )
//	{
//		params = find_json_token(arr, "params");
//
//		int j;
//		char buf[128];
//		uint64_t values[params->num_desc];
//		uint32_t modulusCorrection = 0; // corrects for modulus events in incoming signal
//
//		for(j=0;j<params->num_desc-1;j++)
//		{
//			snprintf(buf, 128, "params[%d]", j);
//			values[j] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf))) + modulusCorrection;
//
//			if( values[j] < values[j-1] && j != 0)
//			{
//				modulusCorrection += ARTEMIS_CLOCK_SPEED_HZ;
//
//				// bump current sample as well
//				values[j] += ARTEMIS_CLOCK_SPEED_HZ;
//			}
//
////			printf("val = %u", values[j]);
//		}
//
//		// last sample is different
//		snprintf(buf, 128, "params[%d]", params->num_desc-1);
//		values[params->num_desc-1] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf)));
//
//		if( handler )
//		{
//			handler->process(values, params->num_desc, 0, 0);
//		}
//	}
//
//	if( method.compare("bs_rq_utc") == 0 )
//	{
//		if( idTok != 0 )
//		{
//			char buf[128];
//			PopTimestamp now = get_microsec_system_time();
//			uint64_t full = now.get_full_secs();
//			uint64_t fracns = now.get_frac_secs()*1000000000;
//			snprintf(buf, 127, "{\"result\":[%lu, %lu], \"id\":%d}", full, fracns, parseUint32_t(FROZEN_GET_STRING(idTok)));
//			buf[127] = '\0';
//			send_rpc(buf, strlen(buf));
//		}
//
//
//
//	}
}


//int b64_decode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );

void PopS3pRPC::packet_rx(std::string b64_serial, uint32_t offset, double clock_correction)
{
//	// the clock correction is 48million cycles of the internal clock divided by the number of cycles for the pps period
//	// this can be viewed as the crystal's tolerance (if we assume pps is 100% accurate)
//	if( clock_correction > 1.8 || clock_correction < .2 )
//	{
//		cout << "Artemis probably doesn't have pps, dropping packet\r\n" << endl;
//	}
//
//	uint32_t maxOffset = ARTEMIS_CLOCK_SPEED_HZ*10; // in units of 10x
//	PopTimestamp now = get_microsec_system_time();
//
////	cout << "now: " << now << endl;
//
//	PopTimestamp m_timestamp_offset;
//	// round system time to nearest second
//	m_timestamp_offset = PopTimestamp(round(now.get_real_secs()));
//
////	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;
//
//	// calculate the fraction component of seconds
//	double fraction = (double) offset / maxOffset;
//
//	// use += operator for lossless addition
//	m_timestamp_offset += PopTimestamp(fraction);
//
////	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;
//
//	// build a "packet" which contains the serial and packet object
//	// we don't use the sink/source built in timestamp stream because we don't have a timestamp for each character in the transmission, just the entire thing
//	boost::tuple<char[20], PopTimestamp> packet;
//	strncpy(get<0>(packet), b64_serial.c_str(), 20);
//	get<1>(packet) = m_timestamp_offset;
//
//
//	// send
//	packets.process(&packet, 1);
}

void PopS3pRPC::forward_packet(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime)
{
	unsigned encodedCount;
	char b64_encoded[b64_length_encoded(size)+1];

	b64_encode(data, size, b64_encoded, &encodedCount);

	// pack in a null so we can %s with printf
	b64_encoded[encodedCount] = '\0';


	char hostname[256];
	int ret = gethostname(hostname, 256);
	if( ret != 0 )
	{
		cout << "couldn't read linux hostname!" << endl;
		strncpy(hostname, "unkown", 256);
	}

	unsigned final_length = 128+encodedCount+256;

	// Guaranteed to be longer than our entire json message
	char buf[final_length];

	unsigned jsonSize = snprintf(buf, final_length, "{\"method\":\"grav_forward\",\"params\":[\"%s\", \"%s\", %" PRIu32 ", %" PRIu64 "]}", hostname, b64_encoded, txTime, pitTxTime );
	send_rpc(buf, jsonSize);
}

void PopS3pRPC::greet_s3p(void)
{
	uint16_t autoinc = rpc_get_autoinc();

	ostringstream os;
	os << "{method:\"grav_boot\",\"params\":[],\"id\":" << autoinc << "}";
	string message = os.str();

	send_rpc(message.c_str(), message.length());
}

void PopS3pRPC::execute_raw(char c, bool reset)
{
	// do nothing
}

}
