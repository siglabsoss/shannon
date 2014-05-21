#include <stdio.h>
#include <stdlib.h>


#include "dsp/prota/popsparsecorrelate.h"
#include "core/basestationfreq.h"
#include "core/util.h"

// 1296 counts is 27us in 48mhz ticks
#define QUICK_SEARCH_STEPS (1296)


int32_t do_comb(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, uint32_t combOffset)
{
	int16_t j,k;
	uint32_t diff;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	short pol; // signal polarity, comb polarity

	xscore = 0; // the "score" of this convolution
	now = start = head = data[0] + combOffset;
	k = 0;
	j = 0;

	nextComb = comb[MIN(k+1, combSize-1)] + start;
	nextSignal = data[j+1];

	// if comb_offset is large enough, we need to skip some edges in the data array, so this scans through edges
	while (now > nextSignal)
	{
		j++;
		nextSignal = data[j+1];
	}

	while(j < dataSize && k < combSize )
	{

		// positive if the signals match, negative if mismatch
		pol = ((j&1) == (k&1))?-1:1;

		// it seems like sometimes dma forgets to transfer 1 byte, causing this problem
		if( now >= head )
		{
			// calculate and sum score
			diff = now - head;
			xscore += diff * pol;
		}


		// bump this number to the current edge
		head = now;

		if( nextComb > nextSignal )
		{
			// next event is a signal edge
			j++;
			now = nextSignal;

			// prep for next comparison
			nextSignal = data[j+1];
		}
		else
		{
			// next event is a comb edge
			k++;
			now = nextComb;

			// prep for next comparison
			nextComb = comb[MIN(k+1, combSize-1)] + start;
		}
	}

	return xscore;
}


uint32_t pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut)
{
	uint32_t denseCombLength = comb[combSize-1] - comb[0];
	uint32_t denseDataLength = 0;

	uint16_t i;

	// we are forced to scan through the input data to determine if any modulus events have occurred in order to get a real value for denseDataLength
	for(i = 1; i < dataSize; i++)
	{
		if( data[i] < data[i-1] )
		{
			denseDataLength += ARTEMIS_CLOCK_SPEED_HZ;
		}

		denseDataLength += data[i]-data[i-1];
	}

	if( denseDataLength < denseCombLength )
	{
		printf("dense data size %d must not be less than dense comb size %d\r\n", denseDataLength, denseCombLength);
		//FIXME: this is not an appropriate way of returning an error condition
		return 0;
	}


	int32_t score, scoreLeft, scoreRight; //x(key)score
	int32_t maxScoreQuick = 0, maxScore = 0;
	uint32_t maxScoreOffsetQuick, maxScoreOffset;
	uint32_t scoreOffsetBinSearch, maxScoreOffsetRight;
	uint32_t iterations;
	iterations = denseDataLength - denseCombLength + 1;
	uint32_t combOffset = 0;

	// quick search
	for(combOffset = 0; combOffset < iterations; combOffset += QUICK_SEARCH_STEPS)
	{
		score = do_comb(data, dataSize, comb, combSize, combOffset);

		if( abs(score) > abs(maxScoreQuick) )
		{
			maxScoreQuick = score;
			maxScoreOffsetQuick = combOffset;
		}
	}




//	printf("max: %u %d\r\n", maxScoreOffsetQuick, maxScoreQuick);

	// we've found a peak
	uint32_t searchStep = QUICK_SEARCH_STEPS;

	maxScoreOffset = scoreOffsetBinSearch = maxScoreOffsetQuick;
	maxScore = maxScoreQuick;


	// warmup loop; we only need to do a single comb because the previous one was done in the quick search
	scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1);

	if( abs(maxScoreQuick) > abs(scoreRight) )
	{
		scoreOffsetBinSearch -= searchStep/2;
	}
	else
	{
		scoreOffsetBinSearch += searchStep/2;
	}


	while( searchStep != 1 )
	{
		searchStep /= 2;

		scoreLeft = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch);

		scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1);

		if( abs(scoreLeft) > abs(scoreRight) )
		{
			if( abs(scoreLeft) > abs(maxScore) )
			{
				maxScore = scoreLeft;
				maxScoreOffset = scoreOffsetBinSearch;
			}

			scoreOffsetBinSearch -= searchStep/2;
		}
		else
		{
			if( abs(scoreRight) > abs(maxScore) )
			{
				maxScore = scoreRight;
				maxScoreOffset = scoreOffsetBinSearch+1;
			}

			scoreOffsetBinSearch += searchStep/2;
		}

		if( searchStep == 1 && scoreLeft == scoreRight )
		{
			//FIXME: this condition can be fixed by curve fitting the searched spots
			printf("Flat peak detected, start of frame will be slightly wrong\r\n");
		}
	}

//	printf("Max offset bin:   %u\r\n", maxScoreOffset);

	*scoreOut = maxScore;

	return data[0] + maxScoreOffset;
}

// pass in a data array including the comb
// pass in the sample which is the end of the comb
uint32_t pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert)
{
	uint32_t denseDataLength = 0;

//	startSample -= 4*2640; //FIXME: remove

	uint16_t i;
	int16_t j,k,jp,kp;
	uint32_t diff;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	short pol; // signal polarity, comb polarity
	uint8_t dataByte = 0;


	uint32_t combSize = (dataOutSize*8) + 1;
	uint32_t comb[combSize];

	double baud = 18181.81818;
	int countsPerBit = (1.0/baud) * ARTEMIS_CLOCK_SPEED_HZ;
	for(i=0;i<combSize;i++)
	{
		comb[i] = countsPerBit * i;
//		printf("combx %u\r\n", comb[i]);
	}


	xscore = 0; // the "score" of this convolution
	now = start = head = startSample;
	kp = k = 0;
	j = 0; // don't set jp, we are about to modify j

	nextComb = comb[MIN(k+1, combSize-1)] + start;
	nextSignal = data[j+1];

	// if comb_offset is large enough, we need to skip some edges in the data array, so this scans through edges
	while (now > nextSignal)
	{
		j++;
		nextSignal = data[j+1];
	}

	jp = j;


//	printf("calculated index of %u\r\n", j);

	while(j < dataSize && k < combSize )
	{

		pol = ((jp&1))?-1:1;

		// it seems like sometimes dma forgets to transfer 1 byte, causing this problem
		if( now >= head )
		{
			// calculate and sum score
			diff = now - head;
			xscore += diff * pol;
		}

		// if the previous loop set 'now' to a comb edge, we are ready to record a bit
		if( kp != k )
		{
			//printf("bit was %d (%d %d)\r\n", xscore, k, kp);

			if( xscore > 0 )
			{
				dataByte <<= 1;
				dataByte  |= 1;
			}
			else
			{
				dataByte <<= 1;
			}

			if( k % 8 == 0 )
			{
				if(invert)
				{
					dataByte ^= 0xff;
				}

				dataOut[(k/8)-1] = dataByte;
				printf("data: %02x\r\n", dataByte);
				dataByte = 0;
			}


			xscore = 0;

		}

		kp = k;
		jp = j;



		// bump this number to the current edge
		head = now;

		if( nextComb > nextSignal )
		{
			// next event is a signal edge
			j++;
			now = nextSignal;

			// prep for next comparison
			nextSignal = data[j+1];
		}
		else
		{

			// next event is a comb edge
			k++;
			now = nextComb;

			// prep for next comparison
			nextComb = comb[MIN(k+1, combSize-1)] + start;
		}



	}




	return 0;


//	printf("calculated index of %u\r\n", index);
}


































