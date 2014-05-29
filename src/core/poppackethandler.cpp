#include <iostream>
#include <boost/timer.hpp>

#include "core/poppackethandler.hpp"
#include "core/util.h"
#include "dsp/prota/popsparsecorrelate.h"
#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"


using namespace std;

namespace pop
{


PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint64_t>("PopPacketHandler", 1), rpc(0)
{

}

char dataTx[2] = {0x0f, 0x00};


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


	printf("Score: %d\r\n", scorePrn);
	printf("Start: %d\r\n", prnCodeStart);
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

	printf("\r\ntime %f\r\n", elapsed_time);


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


//		printf("start end %d %d\r\n", start, end);
		printf("prn data start: %d\r\n", prnCodeStart+combDenseLength);


		uint8_t dataRx[8];

//		printf("Bit sync method:\r\n");
//		pop_data_demodulate(data2, size-1, bitSyncStart+bitSyncDenseLength, dataRx, 2, (scorePrn<0?1:0));

		printf("PRN sync method:\r\n");
		shannon_pop_data_demodulate(data2, size-1, prnCodeStart+combDenseLength, dataRx, 8, (scorePrn<0?1:0));

		uint8_t data_decode[2];

		uint32_t data_decode_size;

		decode_ota_bytes(dataRx, 8, data_decode, &data_decode_size);

		printf("----------------\r\n\r\n");

		printf("data: %02x\r\n", data_decode[0]);
		printf("data: %02x\r\n", data_decode[1]);

		uint8_t bitSyncRx[3];

		printf("\r\n\r\n");




//		if( dataRx[0] == 0xf0 )
//		{


			uint32_t lastTime = data2[size-2];
//
			// pit_last_sample_time this is approximately the time of the last sample from the dma

			// this is approximately the pit time of start of frame
			uint64_t pitPrnCodeStart = pitLastSampleTime - ((lastTime - prnCodeStart)*(double)ARTEMIS_PIT_SPEED_HZ/(double)ARTEMIS_CLOCK_SPEED_HZ);

			double txDelta = 0.75;



			// add .75 seconds
			uint32_t txTime = (prnCodeStart + (uint32_t)(ARTEMIS_CLOCK_SPEED_HZ*txDelta)) % ARTEMIS_CLOCK_SPEED_HZ;

			uint64_t pitTxTime = pitPrnCodeStart + (uint32_t)(ARTEMIS_PIT_SPEED_HZ*txDelta);


			if( rpc )
			{

				ota_packet_t packet;
				ota_packet_zero_fill(&packet);
				packet.type = OTA_PACKET_RPC;
				strcpy(packet.data.rpc.name, "xtal_set_pwm_counts");
				packet.data.rpc.p0 = 0x2100;
				packet.data.rpc.p1 = 0;
				packet.data.rpc.p2 = 0;

				ota_packet_prepare_tx(&packet);
				printf("Checksum ok? %d\r\n", ota_packet_checksum_good(&packet)); // this should always be 1

				rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
			}

			dataTx[1]++;

//			printf("last pit %llu\r\n", pitLastSampleTime);
//			printf("calc pit %llu\r\n", pitPrnCodeStart);
//			printf("%f\r\n\r\n", (pitLastSampleTime-pitPrnCodeStart)/19200000.0);
//
//			printf("code start %lu\r\n", prnCodeStart);
//			printf("last  samp %lu\r\n", lastTime);
//			printf("%f\r\n\r\n", (lastTime-prnCodeStart)/48000000.0);
//
//			printf("%lu\r\n", (txTime - prnCodeStart));
//			printf("%f\r\n\r\n", (txTime - prnCodeStart)/48000000.0);
//
//			printf("%llu    (%llu - %llu)\r\n", (pitTxTime - pitPrnCodeStart), pitTxTime, pitPrnCodeStart);
//			printf("%f\r\n\r\n", (pitTxTime - pitPrnCodeStart)/19200000.0);






			printf("\r\n");
//		}

	}
	else
	{
		printf("prnCodeStart was 0!!\r\n");
	}

//	printf("\r\nMaxScore: %u\r\n", prnCodeStart);

}


} //namespace

