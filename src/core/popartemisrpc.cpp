#include <iostream>
#include <string>
#include <stdint.h>

#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"
#include "b64/b64.h"
#include "core/util.h"
#include "core/utilities.hpp"



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



PopArtemisRPC::PopArtemisRPC(PopFabric *f, std::string a) : PopJsonRPC(0), handler(0), basestation_boot(0), attached_uuid(a), fabric(f)
{
	if( fabric )
	{
		fabric->set_receive_function(boost::bind(&PopArtemisRPC::fabric_rx, this, _1, _2, _3));
	}
}

// call this from main() after all functions are setup to test data demodulation
//FIXME: remove in final version
void PopArtemisRPC::mock(void)
{

	uint64_t values[] = {36366862, 36368974, 36370702, 36372046, 36402190, 36403918, 36407182, 36408718, 36417934, 36419278, 36423694, 36425230, 36428302, 36430414, 36433102, 36446734, 36453070, 36468430, 36472846, 36488590, 36493198, 36509902, 36514702, 36531598, 36535822, 36552910, 36556750, 36574222, 36577870, 36595918, 36599182, 36603790, 36944398, 37163086, 37202830, 37359694, 37414414, 37533070, 37557454, 37602190, 37604686, 37634062, 37735822, 37797070, 37841422, 37855246, 37984654, 38009038, 38085646, 38124814, 38248462, 38299342, 38400718, 38483662, 38522254, 38536270, 38559118, 38690062, 38723086, 38768014, 38781838, 38800654, 38844814, 38949070, 39030286, 39037006, 39077902, 39153934, 39473038, 39671374, 39879694, 40015054, 40270030, 40342222, 40408270, 40442062, 40545358, 40590862, 40735246, 40843342, 40899598, 40949326, 41014798, 41024206, 41194702, 41282446, 41337358, 41483278, 41495758, 41720398, 41823118, 41857102, 41892238, 42116110, 42239374, 42437710, 42619918, 42829390, 42984718, 43056526, 43095502, 43203790, 43269838, 43520590, 43739662, 43775182, 43787662, 43827214, 43876366, 43975246, 44171662, 44191630, 44198926, 44354638, 44499982, 44508238, 44530894, 44714062, 44911630, 45073294, 45101902, 45184078, 45349582, 45485326, 45556750, 45590542, 45687886, 45753934, 45776782, 45833806, 46051918, 46176334, 46226446, 46361614, 46484878, 46652110, 46744078, 46847374, 46865422, 46873294, 46876366, 46884238, 46886926, 46907470, 46928782, 46953166, 46991758, 47005582, 47012878, 47058958, 47083534, 47088718, 47133454, 47134798, 47146894, 47148622, 47149966, 47151694, 47158798, 47160526, 47173966, 47176078, 47201806, 47204110, 47216014, 47217742, 47227918, 47231566, 47237134, 47239438, 47242126, 47243470, 47245198, 47246734, 47252494, 47254606, 47258446, 47260750, 47268046, 47270158, 47280526, 47282062, 47287822, 47289934, 47297038, 47299150, 47304718, 47306830, 47317198, 47319502, 47326606, 47328334, 47332750, 47334478, 47340814, 47343694, 47349262, 47351374, 47357326, 47359054, 47366158, 47368270, 47381518, 47383438, 47392654, 47394382, 47397262, 47398990, 47403022, 47405134, 47409166, 47411470, 47416846, 47418766, 47426254, 47427982, 47442958, 47445070, 47451406, 47452750, 47458510, 47460814, 47464846, 47466574, 47467918, 47469646, 47474062, 47475790, 47476750, 47478862, 47483086, 47485006, 47489038, 47491534, 47499790, 47501518, 47507854, 47509582, 47515150, 47517262, 47521294, 47523022, 47525902, 47528014, 47539726, 47543374, 47550478, 47552590, 47556622, 47558734, 47563150, 47564686, 47569102, 47571406, 47575054, 47577166, 47582734, 47584846, 47602702, 47604622, 47609230, 47610958, 47619982, 47623246, 47629198, 47630734, 47631886, 47633998, 47641102, 47643214, 47659534, 47661646, 47662606, 47664334, 47667214, 47670670, 47676430, 47680078, 47682958, 47684686, 47685646, 47687758, 47694862, 47696974, 47699470, 47701390, 47712142, 47713870, 47715214, 47716558, 47718286, 47721550, 47733262, 47735374, 47742478, 47744590, 47745742, 47747278, 47753038, 47755342, 47759950, 47763022, 47769358, 47770702, 47776462, 47779918, 47785678, 47787598, 47788366, 47790478, 47796238, 47798350, 47805454, 47809294, 47813518, 47815246, 47835214, 47836750, 47839246, 47841358, 47848462, 47851918, 47861134, 47862862, 47863630, 47865742, 47876110, 47878222, 47879182, 47881294, 47893006, 47895118, 47902606, 47904334, 47905294, 47907406, 47919118, 47921230, 47928334, 47930446, 47934862, 47936590, 47943694, 47945806, 47948302, 47950606, 47962318, 47964622, 47971726, 47973454, 47986702, 47990542, 47993230, 47996494, 48005134, 48007246, 48013006, 48014926, 48020494, 48022606, 48026638, 48028558, 48039310, 48041038, 48042382, 48044110, 48059086, 48061198, 48069646, 48071758, 48080206, 48082510, 48094222, 48096526, 48123406, 48124942, 48128014, 48130126, 48134350, 48136654, 48138958, 48141262, 48144910, 48147022, 48149710, 48151630, 48167182, 48168526, 48174286, 48176590, 48181582, 48183694, 48189646, 48191566, 48197134, 48198862, 48201742, 48203854, 48211342, 48213070, 48219022, 48220750, 48229390, 48231502, 48233998, 48236110, 48243214, 48247246, 48250894, 48253006, 48255502, 48257614, 48263758, 48265294, 48271054, 48273166, 48278926, 48280462, 48281614, 48283726, 48289678, 48291406, 48296974, 48299086, 48306382, 48308494, 48321550, 48323662, 48330766, 48332878, 48339022, 48340558, 48346318, 48348430, 48357262, 48358990, 48360334, 48362062, 48366094, 48368206, 48372430, 48375886, 48379918, 48381838, 48386446, 48388558, 48395662, 48397390, 48406030, 48407566, 48412366, 48414478, 48424846, 48426382, 48441358, 48443470, 48452302, 48454414, 48458254, 48460366, 48468814, 48471118, 48476686, 48478798, 48487438, 48489742, 48492238, 48494542, 48501646, 48503374, 48515086, 48517198, 48521230, 48523342, 48539470, 48541582, 48544078, 48546382, 48553486, 48555790, 48561166, 48563086, 48581134, 48583246, 48587278, 48591118, 48598030, 48600526, 48610126, 48612430, 48613390, 48615502, 48618382, 48620110, 48624142, 48626446, 48638158, 48640462, 48648910, 48650830, 48665614, 48667726, 48676558, 48678286, 48684046, 48686158, 48691726, 48695374, 48706318, 48707662, 48713422, 48715342, 999999999999};


	if( handler )
	{
		handler->process(values, ARRAY_LEN(values), 0, 0);
	}
}

void PopArtemisRPC::fabric_rx(std::string to, std::string from, std::string msg)
{
	cout << "(PopArtemisRPC) to: " << to << " from: " << from << " msg: " << msg << endl;


	// The fabric that PopArtemisRPC uses handles the directly attached Artemis in basestation mode, as well as all of the OTA devices
	if(to.compare(attached_uuid) == 0)
	{
		send_rpc(msg.c_str(), msg.length());
	}
	else
	{
		ota_packet_t packet;
		ota_packet_zero_fill(&packet);

//		ostringstream os;
//		os << "{\"result\":[";
//		for( unsigned i = 0; i < chosen; i++ )
//		{
//			if( i != 0 )
//			{
//				os << ",";
//			}
//			os << slots[i];
//		}
//		os << "],\"id\":" << original_id << "}";
//
		snprintf(packet.data, sizeof(packet.data), "%s", msg.c_str()); // lazy way to cap length
		ota_packet_prepare_tx(&packet);
//
		puts(packet.data);

		if( handler )
		{
			handler->enqueue_packet(to, packet);
		}
//
//		packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
	}


}


void PopArtemisRPC::execute_rpc(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
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

		int j;
		char buf[128];
		uint64_t values[params->num_desc];
		uint32_t modulusCorrection = 0; // corrects for modulus events in incoming signal

		for(j=0;j<params->num_desc-1;j++)
		{
			snprintf(buf, 128, "params[%d]", j);
			values[j] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf))) + modulusCorrection;

			if( values[j] < values[j-1] && j != 0)
			{
				modulusCorrection += ARTEMIS_CLOCK_SPEED_HZ;

				// bump current sample as well
				values[j] += ARTEMIS_CLOCK_SPEED_HZ;
			}

//			printf("val = %u", values[j]);
		}

		// last sample is different
		snprintf(buf, 128, "params[%d]", params->num_desc-1);
		values[params->num_desc-1] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf)));

		if( handler )
		{
			handler->process(values, params->num_desc, 0, 0);
		}
	}

	if( method.compare("bs_rq_utc") == 0 )
	{
		if( idTok != 0 )
		{
			char buf[128];
			PopTimestamp now = get_microsec_system_time();
			uint64_t full = now.get_full_secs();
			uint64_t fracns = now.get_frac_secs()*1000000000;
			snprintf(buf, 127, "{\"result\":[%lu, %lu], \"id\":%d}", full, fracns, parseUint32_t(FROZEN_GET_STRING(idTok)));
			buf[127] = '\0';
			send_rpc(buf, strlen(buf));
		}
	}

	if( method.compare("basestation_boot") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING)
		{
			attached_uuid = FROZEN_GET_STRING(p0);
			basestation_boot = 1;
		}
	}
}

void PopArtemisRPC::execute_result(const struct json_token *resultTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str)
{
//	cout << "got result" << str << endl;
	fabric->send("noc", str);
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

	uint32_t maxOffset = ARTEMIS_CLOCK_SPEED_HZ*10; // in units of 10x
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


void PopArtemisRPC::packet_tx(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime)
{
	unsigned encodedCount;
	// b64_length_encoded() tells us the worst case size for the b64 string, we need 1 more char
	char b64_encoded[b64_length_encoded(size)+1];

	// b64 encode data
	b64_encode(data, size, b64_encoded, &encodedCount);

	// pack in a null so we can %s with printf
	b64_encoded[encodedCount] = '\0';

	// Guaranteed to be longer than our entire json message
	char buf[64+encodedCount];

	unsigned jsonSize = snprintf(buf, 64+encodedCount, "{\"method\":\"tx\",\"params\":[\"%s\", %" PRIu32 ", %" PRIu64 "]}", b64_encoded, txTime, pitTxTime );
	send_rpc(buf, jsonSize);
}

void PopArtemisRPC::set_role_base_station()
{
	static const char RPC_STRING[] = "{ \"method\": \"set_role_base_station\", \"params\": [] }";

	// Subtract one from the string size to exclude the trailing '\0' character.
	send_rpc(RPC_STRING, sizeof(RPC_STRING) - 1);
}


void PopArtemisRPC::send_reset()
{
	static const char RPC_STRING[] = "{ \"method\": \"cpu_reset\", \"params\": [] }";

	// Subtract one from the string size to exclude the trailing '\0' character.
	send_rpc(RPC_STRING, sizeof(RPC_STRING) - 1);
}

// have we received a "basestation_boot" message?
int PopArtemisRPC::received_basestation_boot()
{
	return basestation_boot;
}

void PopArtemisRPC::execute_csv(std::string str)
{
	unsigned len = str.length();
	if( len == 0 )
		return;

	// nothing to call process on
	if( !handler )
		return;

	static uint64_t accumulator = 0;

	// make a mutable cstring from the std string because we are going to use strtok
	// http://stackoverflow.com/questions/7352099/stdstring-to-char
	char *csv = new char[str.length() + 1];
	strcpy(csv, str.c_str());


	uint32_t number;
	char *state;
	char *token;
	char seps[] = ",";


	token = strtok_r_single( csv, seps, &state );

	while( token != NULL )
	{
		number = parseHexNumber<uint32_t>(token);
//		printf("%s\r\n", token);

		accumulator += number;

		accumulator %= 0xffffffff;

		edges.process(&accumulator, 1);

		token = strtok_r_single( NULL, seps, &state );
	}



//	printf("\r\n");



//	cout << "csv: " << str << endl;
//
//
//	for(int i = 0; i < str.length(); i++)
//	{
//		char c = str.c_str()[i];
//		if( c == '\n' || c == '\r' )
//		{
//			cout << "new line";
//		}
//
//	}


	delete [] csv;
}

















}
