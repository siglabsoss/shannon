#include <iostream>
#include <boost/timer.hpp>

#include "core/poppackethandler.hpp"
#include "core/util.h"
#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"
#include "core/utilities.hpp"


using namespace std;

namespace pop
{


PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint64_t>("PopPacketHandler", 1), rpc(0)
{

}

void PopPacketHandler::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], char *str, uint32_t txTime, uint64_t pitTxTime)
{
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *params, *p0, *p1, *p2;

	int32_t original_id = -1;

	if( idTok )
	{
		original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
	}


	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
//			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}

	if( method.compare("utc_rq") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			cout << "Serial: " << FROZEN_GET_STRING(p0) << endl;


			// Traditionally all ota packets are replied with a "tx" rpc here.
			// but in this case we want the basestation to behave smart, so we actually send an rpc

			int32_t original_id = -1;

			if( idTok )
			{
				original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
			}

			char buf[128];

			// we encapsulate the original rpc id so that the basestation can correctly reply
			snprintf(buf, 128, "{\"method\":\"bs_send_utc_reply\",\"params\":[%d, %d, %ld]}", original_id, txTime, pitTxTime);

			printf("\r\n");
			puts(buf);

			rpc->send_rpc(buf, strlen(buf));
		}
	}

	if( method.compare("slot_rq") == 0  && original_id != -1 )
	{
//		p0 = find_json_token(arr, "params[0]");
//		if( p0 && p0->type == JSON_TYPE_STRING )
//		{
//			ota_packet_t packet;
//			ota_packet_zero_fill(&packet);
//			snprintf(packet.data, sizeof(packet.data), "{\"result\":[%d],\"id\":%d}", 1, original_id);
//			ota_packet_prepare_tx(&packet);
//
//			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
//		}

		s3p->forward_packet(str, strlen(str), txTime, pitTxTime);
	}

	if( method.compare("poll") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			ota_packet_t packet;
			ota_packet_zero_fill(&packet);
			snprintf(packet.data, sizeof(packet.data), "{}");
			ota_packet_prepare_tx(&packet);

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
		}
	}



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
}

// most of this was copied from popjsonrpc.cpp
void PopPacketHandler::process_ota_packet(ota_packet_t* p, uint32_t txTime, uint64_t pitTxTime)
{
	const char *json = p->data;

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *idTok = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		cout << "problem with json string (" <<  json << ")" << endl;
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

	execute(methodTok, paramsTok, idTok, arr, p->data, txTime, pitTxTime);
}


void PopPacketHandler::process(const uint64_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	cout << "got " << size << " samples" << endl;

	uint64_t pitLastSampleTime = data[size-1];


	uint32_t comb[] = {0, 343200, 559680, 601920, 755040, 813120, 929280, 955680, 997920, 1003200, 1029600, 1135200, 1193280, 1240800, 1251360, 1383360, 1404480, 1483680, 1520640, 1647360, 1694880, 1800480, 1879680, 1921920, 1932480, 1958880, 2085600, 2122560, 2164800, 2180640, 2196480, 2244000, 2344320, 2428800, 2434080, 2476320, 2550240, 2872320, 3067680, 3278880, 3410880, 3669600, 3738240, 3806880, 3838560, 3944160, 3986400, 4134240, 4239840, 4297920, 4345440, 4414080, 4419360, 4593600, 4678080, 4736160, 4878720, 4894560, 5116320, 5221920, 5253600, 5290560, 5512320, 5639040, 5834400, 6019200, 6225120, 6383520, 6452160, 6494400, 6600000, 6668640, 6916800, 7138560, 7170240, 7186080, 7223040, 7275840, 7370880, 7571520, 7587360, 7597920, 7751040, 7898880, 7904160, 7930560, 8110080, 8310720, 8469120, 8500800, 8580000, 8748960, 8880960, 8954880, 8986560, 9086880, 9150240, 9176640, 9229440, 9451200, 9572640, 9625440, 9757440, 9884160, 10047840, 10142880, 10243200};
	uint32_t combDenseLength = comb[ARRAY_LEN(comb)-1];

	// Full comb + bitsync
	// [0, 343200, 559680, 601920, 755040, 813120, 929280, 955680, 997920, 1003200, 1029600, 1135200, 1193280, 1240800, 1251360, 1383360, 1404480, 1483680, 1520640, 1647360, 1694880, 1800480, 1879680, 1921920, 1932480, 1958880, 2085600, 2122560, 2164800, 2180640, 2196480, 2244000, 2344320, 2428800, 2434080, 2476320, 2550240, 2872320, 3067680, 3278880, 3410880, 3669600, 3738240, 3806880, 3838560, 3944160, 3986400, 4134240, 4239840, 4297920, 4345440, 4414080, 4419360, 4593600, 4678080, 4736160, 4878720, 4894560, 5116320, 5221920, 5253600, 5290560, 5512320, 5639040, 5834400, 6019200, 6225120, 6383520, 6452160, 6494400, 6600000, 6668640, 6916800, 7138560, 7170240, 7186080, 7223040, 7275840, 7370880, 7571520, 7587360, 7597920, 7751040, 7898880, 7904160, 7930560, 8110080, 8310720, 8469120, 8500800, 8580000, 8748960, 8880960, 8954880, 8986560, 9086880, 9150240, 9176640, 9229440, 9451200, 9572640, 9625440, 9757440, 9884160, 10047840, 10142880, 10243200, 10264320, 10269600, 10274880, 10280160, 10285440, 10288080, 10290720, 10293360, 10296000, 10298640, 10301280, 10303920, 10306560]



	size_t i,j;
	uint32_t prnCodeStart, bitSyncStart;

	int32_t scorePrn, scoreBitSync;


	uint32_t data2[size-1];

//	printf("\r\n");
	for(i=0;i<(size-1);i++)
	{
		data2[i] = (uint32_t)data[i];
//		printf("%u, ", data2[i]);
	}
//	printf("\r\n\r\n");


	boost::timer t; // start timing

	prnCodeStart = shannon_pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);


//	printf("Score: %d\r\n", scorePrn);
//	printf("Start: %d\r\n", prnCodeStart);
//	if( abs(scorePrn) < )

	double elapsed_time = t.elapsed();

	if( prnCodeStart == 0 || elapsed_time > 4.0 )
	{
		printf("\r\n");

		for(j = 0; j<size-1;j++)
		{
			printf("%u, ", data2[j]);
		}

//		pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);

	}

//	printf("\r\ntime %f\r\n", elapsed_time);


	if( prnCodeStart != 0 )
	{
		short flag1 = 0, flag2 = 0;
		for(i = 1; i < size-1; i++)
		{
			if( data2[i] > (prnCodeStart+combDenseLength) && !flag1 )
			{
				flag1 = 1;
			}
		}

		if( !flag1 )
		{
			printf("\r\n");
			for(i=0;i<(size-1);i++)
			{
				data2[i] = (uint32_t)data[i];
				printf("%u, ", data2[i]);
			}
			printf("\r\n\r\n");

			printf("data was not longer than comb %d %d\r\n", data2[size-1], (prnCodeStart+combDenseLength) );

			return;
		}




		uint32_t lastTime = data2[size-2];
		//
		// pit_last_sample_time this is approximately the time of the last sample from the dma

		// this is approximately the pit time of start of frame
		uint64_t pitPrnCodeStart = pitLastSampleTime - ((lastTime - prnCodeStart)*(double)ARTEMIS_PIT_SPEED_HZ/(double)ARTEMIS_CLOCK_SPEED_HZ);

		printf("PIT start: %lu\r\n", pitPrnCodeStart);

		double txDelta = 0.75;



		// add .75 seconds
		uint32_t txTime = (prnCodeStart + (uint32_t)(ARTEMIS_CLOCK_SPEED_HZ*txDelta)) % ARTEMIS_CLOCK_SPEED_HZ;

		uint64_t pitTxTime = pitPrnCodeStart + (uint32_t)(ARTEMIS_PIT_SPEED_HZ*txDelta);



















		ota_packet_t rx_packet;

		unsigned peek_length = 4; // how many bytes do we need to find the size?
		unsigned peek_length_encoded = ota_length_encoded(peek_length);

		uint8_t data_rx[peek_length_encoded];


		shannon_pop_data_demodulate(data2, size-1, prnCodeStart+combDenseLength, data_rx, peek_length_encoded, (scorePrn<0?1:0));

		uint8_t data_decode[peek_length];

		uint32_t data_decode_size;

		decode_ota_bytes(data_rx, peek_length_encoded, data_decode, &data_decode_size);


		//	printf("data: %02x\r\n", data_decode[0]);
		//	printf("data: %02x\r\n", data_decode[1]);
		//	printf("data: %02x\r\n", data_decode[2]);
		//	printf("data: %02x\r\n", data_decode[3]);
		//

		ota_packet_zero_fill(&rx_packet);
		memcpy(&rx_packet, data_decode, peek_length);
		uint16_t packet_size = MIN(rx_packet.size, ARRAY_LEN(rx_packet.data)-1);

		int packet_good = 0;
		int j;

		// search around a bit till the checksum matches up.  This is our "bit sync"
		for(j = -2400; j < 2400; j+=300)
		{
			uint16_t decode_remainig_size = MAX(0, packet_size);
			unsigned remaining_length = ota_length_encoded(decode_remainig_size);
			uint8_t data_rx_remaining[remaining_length];
			shannon_pop_data_demodulate(data2, size-1, prnCodeStart+combDenseLength+j, data_rx_remaining, remaining_length, (scorePrn<0?1:0));
			uint8_t data_decode_remaining[decode_remainig_size];
			uint32_t data_decode_size_remaining;
			decode_ota_bytes(data_rx_remaining, remaining_length, data_decode_remaining, &data_decode_size_remaining);

			// the null terminated character is not transmitted
			ota_packet_zero_fill_data(&rx_packet);
			memcpy(((uint8_t*)&rx_packet), data_decode_remaining, decode_remainig_size);

			if(ota_packet_checksum_good(&rx_packet))
			{
				packet_good = 1;
				break;
			}
		}

		if( !packet_good )
		{
			printf("Bad packet checksum\r\n");
			return;
		}

		if(rx_packet.data[ARRAY_LEN(rx_packet.data)-1] != '\0' )
		{
			printf("Packet c-string is not null terminated\r\n");
			return;
		}

		printf("Packet says: %s\r\n", rx_packet.data);

		if( rpc )
		{
			process_ota_packet(&rx_packet, txTime, pitTxTime);
		}
		else
		{
			printf("Rpc pointer not set, skipping json parse\r\n");
		}











		if( 0 && rpc )
		{

			ota_packet_t packet;
			ota_packet_zero_fill(&packet);
			//strcpy(packet.data, "{\"method\":\"xtal_set_pwm_counts\",\"params\":[8448]}");
			strcpy(packet.data, "{\"method\":\"mkw01_print_regs\",\"params\":[]}");
			ota_packet_prepare_tx(&packet);
			//				printf("Checksum ok? %d\r\n", ota_packet_checksum_good(&packet)); // this should always be 1

			// the null terminated character is not transmitted

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
		}





		printf("\r\n");

	}
	else
	{
		printf("prnCodeStart was 0!!\r\n");
	}

//	printf("\r\nMaxScore: %u\r\n", prnCodeStart);

}


} //namespace

