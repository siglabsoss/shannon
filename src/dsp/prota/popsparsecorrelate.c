#include <stdio.h>
#include <stdlib.h>

#include "dsp/prota/popsparsecorrelate.h"
#include "core/util.h"

// 1296 counts is 27us in 48mhz ticks
#define QUICK_SEARCH_STEPS (300)


int32_t do_comb(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, uint32_t combOffset)
{
	int16_t j,k;
	uint32_t diff;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	uint32_t modulusCorrection = 0; // corrects for modulus events in incoming signal
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

			// data modulous detected, carry this value forward for the rest of the xcorr
			if( data[j+1] < data[j] )
			{
				modulusCorrection += 48000000;
			}

			// prep for next comparison
			nextSignal = data[j+1] + modulusCorrection;
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


uint32_t pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* score)
{
	uint32_t denseCombLength = comb[combSize-1] - comb[0];
	uint32_t denseDataLength = 0;

	uint16_t i;

	// we are forced to scan through the input data to determine if any modulus events have occurred in order to get a real value for denseDataLength
	for(i = 1; i < dataSize; i++)
	{
		if( data[i] < data[i-1] )
		{
			denseDataLength += 48000000;
		}

		denseDataLength += data[i]-data[i-1];
	}

	if( denseDataLength < denseCombLength )
	{
		printf("dense data size %d must not be less than dense comb size %d\r\n", denseDataLength, denseCombLength);
		//FIXME: this is not an appropriate way of returning an error condition
		return 0;
	}

//	printf("\r\n\r\n");

	/*
	 * Matlab's xcorr(x,y)
	 * takes y at the leftmost part of x (so that only 1 sample overlaps) and then slides y forwards
	 *
	 *
	 *
	 *
	 *
	 * xcorr(toDense([0 1     3     4     5     7     9]), toDense([0,1,2,4,6]))
	 *
	 *  1    -1    -1     1    -1     1     1    -1    -1     1
	 *
	 *  1    -1     1     1    -1    -1     1
	 */

	int32_t xscore; //x(key)score
	uint32_t maxScore = 0;
	uint32_t maxScoreOffset;
	uint32_t iterations;
	iterations = denseDataLength - denseCombLength + 1;
	uint32_t combOffset = 0;

	// quick search
	for(combOffset = 0; combOffset < iterations; combOffset += QUICK_SEARCH_STEPS)
	{
		xscore = do_comb(data, dataSize, comb, combSize, combOffset);

		if( abs(xscore) > abs(maxScore) )
		{
			maxScore = xscore;
			maxScoreOffset = combOffset;
		}
	}

	uint32_t combSlowStart = MAX(0,(maxScoreOffset-QUICK_SEARCH_STEPS+1));
	uint32_t combSlowEnd = MIN(iterations,(maxScoreOffset+QUICK_SEARCH_STEPS-1));

	// slow search
	for(combOffset = combSlowStart; combOffset < combSlowEnd; combOffset++)
	{
		xscore = do_comb(data, dataSize, comb, combSize, combOffset);

		// score is ready
		if( abs(xscore) > abs(maxScore) )
		{
			maxScore = xscore;
			maxScoreOffset = combOffset;
		}
	}

	*score = maxScore;

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
	uint32_t modulusCorrection = 0; // corrects for modulus events in incoming signal
	short pol; // signal polarity, comb polarity
	uint8_t dataByte = 0;


	uint32_t combSize = (dataOutSize*8) + 1;
	uint32_t comb[combSize];

	double baud = 18181.81818;
	int countsPerBit = (1.0/baud) * 48000000.0;
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

				dataOut[k/8] = dataByte;
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

			// data modulous detected, carry this value forward for the rest of the xcorr
			if( data[j+1] < data[j] )
			{
				modulusCorrection += 48000000;
			}

			// prep for next comparison
			nextSignal = data[j+1] + modulusCorrection;
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


































