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

void PopPacketHandler::process(const uint64_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	cout << "got " << size << " samples" << endl;

	uint64_t pitLastSampleTime = data[size-1];

	// This is the bit sync pattern for FF,33,55
	uint32_t bitSync[] = {0, 21120, 26400, 31680, 36960, 42240, 44880, 47520, 50160, 52800, 55440, 58080, 60720, 63360};
	uint32_t bitSyncDenseLength = bitSync[ARRAY_LEN(bitSync)-1];



	uint32_t comb[] = {0, 343200, 559680, 601920, 755040, 813120, 929280, 955680, 997920, 1003200, 1029600, 1135200, 1193280, 1240800, 1251360, 1383360, 1404480, 1483680, 1520640, 1647360, 1694880, 1800480, 1879680, 1921920, 1932480, 1958880, 2085600, 2122560, 2164800, 2180640, 2196480, 2244000, 2344320, 2428800, 2434080, 2476320, 2550240, 2872320, 3067680, 3278880, 3410880, 3669600, 3738240, 3806880, 3838560, 3944160, 3986400, 4134240, 4239840, 4297920, 4345440, 4414080, 4419360, 4593600, 4678080, 4736160, 4878720, 4894560, 5116320, 5221920, 5253600, 5290560, 5512320, 5639040, 5834400, 6019200, 6225120, 6383520, 6452160, 6494400, 6600000, 6668640, 6916800, 7138560, 7170240, 7186080, 7223040, 7275840, 7370880, 7571520, 7587360, 7597920, 7751040, 7898880, 7904160, 7930560, 8110080, 8310720, 8469120, 8500800, 8580000, 8748960, 8880960, 8954880, 8986560, 9086880, 9150240, 9176640, 9229440, 9451200, 9572640, 9625440, 9757440, 9884160, 10047840, 10142880, 10243200};
	uint32_t combDenseLength = comb[ARRAY_LEN(comb)-1];



	size_t i,j;
	uint32_t prnCodeStart, bitSyncStart;

	int32_t scorePrn, scoreBitSync;


	uint32_t data2[size-1];

	for(i=0;i<(size-1);i++)
	{
		data2[i] = (uint32_t)data[i];
	}


	boost::timer t; // start timing

	prnCodeStart = pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);

//	printf("Score: %d\r\n", scorePrn);
//	if( abs(scorePrn) < )

	double elapsed_time = t.elapsed();

	if( elapsed_time > 4.0 )
	{
		printf("\r\n");

		for(j = 0; j<size-1;j++)
		{
			printf("%u, ", data2[j]);
		}

//		pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);

	}

	printf("\r\ntime %f\r\n", elapsed_time);

	if( elapsed_time > 0.75 )
	{
		printf("\r\nSkipping TX, xcorr took too long\r\n");
		return;
	}


	//uint32_t cooked[] = {0, 2640, 5280, 7920, 10560, 13200, 15840, 18480, 21120, 23760, 26400, 29040, 31680, 34320, 36960, 39600, 42240, 44880, 47520, 50160, 52800, 55440, 58080, 60720, 63360, 66000, 68640, 71280};


	if( prnCodeStart != 0 )
	{

		uint32_t i, start, end;
		short flag1 = 0, flag2 = 0;
		for(i = 1; i < size-1; i++)
		{
			if( data2[i] > (prnCodeStart+combDenseLength) && !flag1 )
			{
				flag1 = 1;
				start = i-1;
			}

			if( data2[i] > (prnCodeStart+combDenseLength+bitSyncDenseLength) && !flag2 )
			{
				flag2 = 1;
				end = MIN(i+1, size-1);
			}
		}

		if( !flag1 || !flag2 )
		{
			printf("data was not longer than comb + bit sync code %d %d\r\n", data2[size-1], (prnCodeStart+combDenseLength+bitSyncDenseLength) );

//			printf("\r\n");
//			for(j = 0; j<size;j++)
//			{
//				printf("%u, ", data[j]);
//			}
//			printf("\r\n");

			return;
		}


//		printf("start end %d %d\r\n", start, end);

		bitSyncStart = pop_correlate(data2+start, (end-start), bitSync, ARRAY_LEN(bitSync), &scoreBitSync);

//		printf("score2: %d\r\n", scoreBitSync);


		uint8_t dataRx[2];

//		printf("Bit sync method:\r\n");
		pop_data_demodulate(data2, size-1, bitSyncStart+bitSyncDenseLength, dataRx, 2, (scorePrn<0?1:0));

//		if( dataRx[0] == 0xf0 )
//		{


			uint32_t lastTime = data2[size-2];
//
			// pit_last_sample_time this is approximately the time of the last sample from the dma

			// this is approximately the pit time of start of frame
			uint64_t pitPrnCodeStart = pitLastSampleTime - ((lastTime - prnCodeStart)*(double)ARTEMIS_PIT_SPEED_HZ/(double)ARTEMIS_CLOCK_SPEED_HZ);

			double txDelta = 0.75;

			char dataTx[2] = {0x54, 0x42};

			// add .75 seconds
			uint32_t txTime = (prnCodeStart + (uint32_t)(ARTEMIS_CLOCK_SPEED_HZ*txDelta)) % ARTEMIS_CLOCK_SPEED_HZ;

			uint64_t pitTxTime = pitPrnCodeStart + (uint32_t)(ARTEMIS_PIT_SPEED_HZ*txDelta);


			if( rpc )
			{
				rpc->packet_tx(dataTx, 2, txTime, pitTxTime);
			}

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

