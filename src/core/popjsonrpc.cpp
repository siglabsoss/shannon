#include "core/popjsonrpc.hpp"
#include "b64/b64.h"

#include <iostream>
#include <string>


using namespace std;



namespace pop
{



// functions
void rcp_log(std::string log)
{
	cout << log << endl;
}


int rpc_count()
{
	static int i = 0;
	cout << "Bump to " << ++i << endl;
	return i;
}

void ppp(std::string p)
{
	cout << p << endl;
}


PopJsonRPC::PopJsonRPC(unsigned notused) : PopSink<unsigned char>("PopJsonRPCSink", 1), rx("PopJsonRPCResponse"), headValid(false)
{
	tx = this;
}

void PopJsonRPC::init() {}

void PopJsonRPC::process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	if( size != 1 ) {
		cout << "Error " << this->get_name() << " may only accept 1 character at a time";
		return;
	}

	char c = data[0];

	//cout << c;

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


void PopJsonRPC::parse()
{
	unsigned len = command.size();
	if( len == 0 )
		return;

	std::string str(command.begin(),command.end());

	const char *json = str.c_str();

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok, *paramsTok, *idTok;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		// skip printing this message for simple newline messages.  if one string matches, it returns 0 which we then multiply
		if( ( str.compare("\r\n") * str.compare("\n") * str.compare("\r") ) != 0)
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

//	std::string method;
	// Search for parameter "bar" and print it's value
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}
	else
	{
//		method = std::string(methodTok->ptr, methodTok->len);
	}


	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	int methodId = -1;
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
//		return;
	}
	else
	{
//		std::string sval = std::string(tok3->ptr, tok3->len);
//		methodId = std::stoi(sval);
//
//		if( methodId < 0 )
//			return;
	}




	execute(methodTok, paramsTok, idTok, arr, str);

}

void PopJsonRPC::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *p0, *p1, *p2;

	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}

//	if( method.compare("count") == 0 )
//	{
//		int ret = rpc_count();
//		respond_int(ret, methodId);
//	}

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
}

void PopJsonRPC::respond_int(int value, int methodId)
{

	std::ostringstream ss;
	//    	ss << "This is " << cs << "!";
	//    	std::cout << ss.str() << std::endl

	ss << "{\"result\":" << value << ", \"error\": null, \"id\": " << methodId << "}";

	std::string str = ss.str();
	unsigned char *buff;

	buff = rx.get_buffer(1);
	buff[0] = '\0';
	rx.process(1);

	// should copy in all the characters but omit the final null
	buff = rx.get_buffer(str.size());
	strncpy((char*)buff, str.c_str(), str.size());
	rx.process(str.size());

	buff = rx.get_buffer(1);
	buff[0] = '\0';
	rx.process(1);
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

void PopJsonRPC::packet_rx(std::string b64_serial, uint32_t offset, double clock_correction)
{
	// the clock correction is 48million cycles of the internal clock divided by the number of cycles for the pps period
	// this can be viewed as the crystal's tolerance (if we assume pps is 100% accurate)
	if( clock_correction > 1.8 || clock_correction < .2 )
	{
		cout << "Artemis probably doesn't have pps, dropping packet\r\n" << endl;
	}

	uint32_t maxOffset = 480000000; // in units of 10x
	PopTimestamp now = get_microsec_system_time();

	cout << "now: " << now << endl;

	PopTimestamp m_timestamp_offset;
	// round system time to nearest second
	m_timestamp_offset = PopTimestamp(round(now.get_real_secs()));

	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;

	double fraction = (double) offset / maxOffset;

	m_timestamp_offset += PopTimestamp(fraction);

	cout << "m_timestamp_offset: " << m_timestamp_offset << endl;

	// build a "packet"
	boost::tuple<char[20], PopTimestamp> packet;// = boost::tuple<(char[20])b64_serial.c_str(), PopTimestamp>(0,m_timestamp_offset);

	strncpy(get<0>(packet), b64_serial.c_str(), 20);
	get<1>(packet) = m_timestamp_offset;
//	packet.


	packets.process(&packet, 1);



	// below we add the radio seconds (which count up since launch) to our offset which doesn't change.
	// at this point the radio seconds are probably about 2.0001
	// we want to subract the whole seconds from our m_timestamp_offset right now (one time) so we can just do a simple add below and get real time
	// ( this uses the constructor to construct a temporary timestamp object holding just N whole seconds. then we use the -= overload to subtract it)
//	m_timestamp_offset -= uhd::time_spec_t(md.time_spec.get_full_secs());


	//                cout << "rounded to base: '" << m_timestamp_offset.get_full_secs() << "' '" <<  m_timestamp_offset.get_frac_secs()<< "' from now of: '" << now.get_full_secs()  << "' '" << now.get_frac_secs() << "'" << endl;



// build a pop timestamp from uhd time + offset.
// the time always applies to sample 0
//PopTimestamp pop_stamp = PopTimestamp(md.time_spec + m_timestamp_offset);








	cout << "in packet_rx: " << b64_serial << endl;
	unsigned encodedCount = b64_serial.length();
	char serialDecoded[encodedCount];
	unsigned decodedCount;
	b64_decode(b64_serial.c_str(), encodedCount, serialDecoded, &decodedCount);


	cout << "Serial: ";

	for(unsigned i = 0; i<decodedCount;i++)
	{
//		cout <<  serialDecoded[i];
		printf("%02x", serialDecoded[i]);
	}

	cout << endl;

}

}
