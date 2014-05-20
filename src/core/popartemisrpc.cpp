#include <iostream>
#include <string>
#include <stdint.h>

#include "core/popartemisrpc.hpp"
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



PopArtemisRPC::PopArtemisRPC(unsigned notused) : PopJsonRPC(0), handler(0)
{
}

// call this from main() after all functions are setup to test data demodulation
//FIXME: remove in final version
void PopArtemisRPC::mock(void)
{
	uint64_t values[] = {7041827, 7090595, 7093667, 7109219, 7123427, 7130531, 7144739, 7151075, 7166435, 7172387, 7187747, 7193699, 7207907, 7215011, 7229411, 7235171, 7250723, 7256675, 7272419, 7277987, 7279715, 7623587, 7840163, 7882403, 8035619, 8092643, 8210339, 8235491, 8278115, 8282915, 8310755, 8415203, 8473571, 8520995, 8532131, 8663459, 8685155, 8763683, 8800739, 8927843, 8975459, 9081059, 9160163, 9202595, 9213347, 9239459, 9365987, 9402659, 9445667, 9460835, 9476579, 9523811, 9624803, 9708515, 9714467, 9756899, 9831011, 10152227, 10348643, 10559459, 10691555, 10949603, 11018723, 11087075, 11119907, 11224355, 11267171, 11414435, 11520419, 11577827, 11626403, 11694371, 11700323, 11873699, 11958755, 12016931, 12159587, 12174947, 12397475, 12502115, 12534371, 12571043, 12792803, 12919715, 13115171, 13299491, 13506467, 13663715, 13732835, 13775267, 13881827, 13949027, 14197667, 14418851, 14451683, 14466659, 14503715, 14557475, 14652131, 14852003, 14868323, 14878499, 15032099, 15179363, 15185315, 15211427, 15391331, 15591203, 15750563, 15781283, 15861155, 16029155, 16162211, 16236323, 16267235, 16367075, 16431203, 16457699, 16510883, 16732067, 16853603, 16906019, 17038307, 17165219, 17328995, 17423267, 17524643, 17529251, 17535395, 17540003, 17545571, 17550179, 17555939, 17560547, 17577059, 17579939, 17582051, 17837795, 17840099, 17871779, 17873891, 17879843, 17881571, 17888675, 17890787, 17905763, 17908067, 17916323, 17918435, 17931683, 17933987, 17942435, 17944163, 17967011, 17969315, 17979107, 17981411, 17982371, 17984483, 17997731, 18000035, 18036131, 18038243, 18044195, 18045923, 18050339, 18051875, 18061283, 18062819, 18073187, 18075299, 18084131, 18085859, 18094307, 18096611, 18108515, 18111587, 18113123, 18115427, 18117731, 18119459, 18133283, 18135011, 18145955, 18147299, 18148259, 18150755, 18154403, 18156515, 18159203, 18161123, 18172835, 18174947, 18177443, 18179171, 18186659, 18189923, 18197795, 18199523, 18200291, 18202595, 18205475, 18207203, 18214307, 18216611, 18220451, 18222563, 18228323, 18230627, 18236195, 18239075, 18240803, 18242531, 18247139, 18248675, 18264995, 18266915, 18270947, 18273251, 18278819, 18280931, 18285155, 18287267, 18292643, 18294563, 18303779, 18305315, 18310883, 18313187, 18321827, 18323939, 18329699, 18332003, 18335651, 18337763, 18342179, 18343907, 18346211, 18348515, 18354083, 18357347, 18361763, 18363683, 18367907, 18369827, 18370979, 18373091, 18379043, 18380771, 18389411, 18393251, 18397859, 18399203, 18406307, 18408419, 18412835, 18416099, 18427811, 18429923, 18441827, 18443747, 18450851, 18452963, 18460067, 18462179, 18466403, 18469859, 18472739, 18474083, 18475619, 18477347, 18487523, 18489827, 18493859, 18495587, 18501539, 18503843, 18506147, 18508643, 18512291, 18514403, 18533795, 18535907, 18541475, 18543971, 18555683, 18557411, 18562979, 18565091, 18567587, 18569699, 18586211, 18588131, 18592163, 18594275, 18606371, 18607715, 18610595, 18612515, 18616931, 18619235, 18627491, 18629411, 18633635, 18635747, 18638435, 18640355, 18645923, 18647843, 18653603, 18655715, 18662819, 18664931, 18666275, 18668003, 18679715, 18681827, 18688931, 18690851, 18694883, 18697187, 18701603, 18703331, 18710435, 18712547, 18718115, 18720419, 18727523, 18729827, 18730211, 18732515, 18736931, 18738659, 18744227, 18747875, 18753443, 18757091, 18759971, 18761699, 18768995, 18770915, 18775331, 18777059, 18778019, 18780131, 18787235, 18789155, 18796643, 18798371, 18802979, 18804707, 18807587, 18810851, 18814883, 18816995, 18822947, 18824675, 18831779, 18833699, 18838115, 18840035, 18848675, 18850211, 18852323, 18853859, 18858275, 18860003, 18868835, 18871139, 18874787, 18876899, 18878243, 18879971, 18904163, 18906467, 18919523, 18921443, 18924323, 18926051, 18937763, 18939683, 18945443, 18947555, 18953507, 18955235, 18956195, 18958307, 18966947, 18969059, 18974819, 18977123, 18985379, 18987491, 18990371, 18992099, 18997667, 18999779, 19005347, 19007075, 19011491, 19013603, 19019555, 19021091, 19025315, 19027427, 19031459, 19033571, 19039139, 19041059, 19044131, 19045667, 19053347, 19055075, 19057379, 19059683, 19066787, 19068899, 19073123, 19075043, 19080803, 19083107, 19088099, 19090403, 19094819, 19098083, 19109795, 19113635, 19117475, 19119587, 19130147, 19131875, 19137443, 19139555, 19140515, 19142051, 19144355, 19147043, 19160867, 19162595, 19169699, 19171811, 19172771, 19174883, 19181987, 19184291, 19186787, 19189091, 19193123, 19194851, 19201955, 19205603, 19211171, 19213283, 19228451, 19230179, 19240355, 19242467, 19246691, 19248611, 19252643, 19254947, 19259171, 19260899, 19264739, 19266851, 19271459, 19273187, 19273955, 19276259, 19294115, 19296227, 19303331, 19306979, 19312547, 19314659, 19321763, 19325027, 19329443, 19331555, 19334051, 19336163, 19342115, 19343843, 19352483, 19354595, 19366499, 19368227, 19375331, 19377635, 19382051, 19383779, 19392803, 19394531, 19400867, 19402211, 19404899, 19406819, 19412771, 19414499, 19418531, 19420643, 19424867, 19426787, 19429283, 19431587, 19440227, 19442147, 19449251, 19451363, 19469411, 19471523};


	if( handler )
	{
		handler->process(values, ARRAY_LEN(values), 0, 0);
	}
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
		values[params->num_desc-1] = parseUint64_t(FROZEN_GET_STRING(find_json_token(arr, buf))) + modulusCorrection;

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

	// leading null
	this->tx.process("\0", 1);

	unsigned jsonSize = snprintf(buf, 64+encodedCount, "{\"method\":\"tx\",\"params\":[\"%s\", %" PRIu32 ", %" PRIu64 "]}", b64_encoded, txTime, pitTxTime );
	this->tx.process(buf, jsonSize);

	this->tx.process("\0", 1);
}

}
