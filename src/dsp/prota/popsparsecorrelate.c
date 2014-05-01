#include <stdio.h>
#include <stdlib.h>

#include "dsp/prota/popsparsecorrelate.h"
#include "core/util.h"

#define DMA_SAMP(x) (0)

int32_t pop_correlate(uint32_t* data, uint16_t dataSize, uint32_t* comb, uint32_t combSize)
{
	printf("hi\r\n");

	uint32_t denseDataLength = data[dataSize-1] - data[0];
	uint32_t denseCombLength = comb[combSize-1] - comb[0];

	if( denseDataLength < denseCombLength )
	{
		printf("dense data size %d must not be less than dense comb size %d\r\n", denseDataLength, denseCombLength);
	}


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



	uint32_t time = 0;

	int16_t j,k;
	uint32_t diff;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	uint32_t iterations;
	int debug;
	char pol; // signal polarity, comb polarity


	iterations = denseDataLength - denseCombLength + 1;
	uint32_t comb_offset = 0;

	for(comb_offset = 0; comb_offset < iterations; comb_offset++)
	{

		xscore = 0; // the "score" of this convolution
		now = start = head = data[0] + comb_offset;
		k = 0;
		j = 0;

		nextComb = comb[MIN(k+1, combSize-1)] + start;
		nextSignal = data[j+1];





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

		printf("Score: %d\r\n", xscore);

	}

}




































