#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>

//#include "phy/popsparsecorrelate.h"
#include "dsp/prota/popsparsecorrelate.h"

#ifdef POPWI_PLATFORM_ARTEMIS
#include "phy/radio.h"
#include "hal/cpu.h"
#include "hal/dma.h"
#include "hal/pit.h"
#else
#include "core/basestationfreq.h"
#include "core/util.h"
#endif



#ifndef POPWI_PLATFORM_ARTEMIS
const uuid_t zero_uuid = {{{0}}};
#endif


// Putting const here BREAKS this array!?? (it gets stored at 0x0)
//uint32_t signal_comb[] = {0, 2640, 7920, 13200, 21120, 23760, 34320, 36960, 39600, 42240, 55440, 58080, 60720, 71280, 79200, 87120, 89760, 92400, 95040, 100320, 102960, 118800, 126720, 129360, 132000, 134640, 137280, 139920, 145200, 161040, 166320, 179520, 182160, 192720, 195360, 198000, 205920, 211200, 213840, 216480, 240240, 245520, 248160, 253440, 261360, 264000, 266640, 274560, 277200, 279840, 282480, 285120, 293040, 295680, 298320, 306240, 308880, 316800, 322080, 324720, 332640, 335280, 340560, 351120, 361680, 364320, 366960, 369600, 372240, 374880, 377520, 380160, 382800, 385440, 388080, 390720, 403920, 409200, 414480, 417120, 422400, 435600, 438240, 443520, 448800, 454080, 467280, 472560, 477840, 480480, 483120, 485760, 488400, 491040, 493680, 498960, 514800, 525360, 530640, 535920, 538560, 541200, 543840, 549120, 551760, 559680, 564960, 575520, 578160, 583440, 586080, 588720, 596640, 617760, 620400, 623040, 636240, 638880, 646800, 654720, 657360, 660000, 662640, 665280, 667920, 675840, 681120, 704880, 712800, 723360, 728640, 731280, 736560, 739200, 741840, 744480, 747120, 749760, 757680, 762960, 765600, 770880, 776160, 778800, 781440, 784080, 789360, 792000, 799920, 802560, 805200, 807840, 821040, 831600, 847440, 852720, 858000, 863280, 865920, 876480, 881760, 884400, 887040, 889680, 892320, 897600, 910800, 921360, 926640, 929280, 931920, 937200, 942480, 945120, 947760, 953040, 958320, 963600, 966240, 968880, 976800, 982080, 987360, 990000, 1000560, 1003200, 1005840, 1008480, 1011120, 1029600, 1045440, 1056000};
// Chirp at 18181.81818
//uint32_t signal_comb[] = {0, 84480, 168960, 253440, 337920, 422400, 506880, 591360, 675840, 760320, 844800, 929280, 1013760, 1098240, 1182720, 1267200, 1351680, 1436160, 1520640, 1605120, 1689600, 1774080, 1858560, 1943040, 2027520, 2112000, 2196480, 2280960, 2365440, 2449920, 2534400, 2618880, 2703360, 2787840, 2872320, 2956800, 3041280, 3125760, 3210240, 3231360, 3252480, 3273600, 3294720, 3315840, 3336960, 3358080, 3379200, 3400320, 3421440, 3442560, 3463680, 3484800, 3505920, 3527040, 3548160};
uint32_t signal_comb[] = {0, 343200, 559680, 601920, 755040, 813120, 929280, 955680, 997920, 1003200, 1029600, 1135200, 1193280, 1240800, 1251360, 1383360, 1404480, 1483680, 1520640, 1647360, 1694880, 1800480, 1879680, 1921920, 1932480, 1958880, 2085600, 2122560, 2164800, 2180640, 2196480, 2244000, 2344320, 2428800, 2434080, 2476320, 2550240, 2872320, 3067680, 3278880, 3410880, 3669600, 3738240, 3806880, 3838560, 3944160, 3986400, 4134240, 4239840, 4297920, 4345440, 4414080, 4419360, 4593600, 4678080, 4736160, 4878720, 4894560, 5116320, 5221920, 5253600, 5290560, 5512320, 5639040, 5834400, 6019200, 6225120, 6383520, 6452160, 6494400, 6600000, 6668640, 6916800, 7138560, 7170240, 7186080, 7223040, 7275840, 7370880, 7571520, 7587360, 7597920, 7751040, 7898880, 7904160, 7930560, 8110080, 8310720, 8469120, 8500800, 8580000, 8748960, 8880960, 8954880, 8986560, 9086880, 9150240, 9176640, 9229440, 9451200, 9572640, 9625440, 9757440, 9884160, 10047840, 10142880, 10243200};
#define COMB_THRESH_WEAK ((unsigned) (COMB_LENGTH * 0.5) )
#define COMB_THRESH ((unsigned) (COMB_LENGTH * 0.55) )
#define COMB_SIZE (ARRAY_LEN(signal_comb))




#ifdef POPWI_PLATFORM_ARTEMIS

#define DATA_SAMPLE(x) DMA2_3_SAMP((x))

#define FN_ATTRIBUTES __attribute__((section(".ram")))

// search this many steps +/- around the guess for Artemis
#define GUESS_ERROR (10000)

#else
#define DATA_SAMPLE(x) data[x]
#define FN_ATTRIBUTES
#endif


// forward declare
FN_ATTRIBUTES uint32_t core_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut, int32_t guess);
FN_ATTRIBUTES uint32_t core_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);


// These wrappers are platform specific
#ifdef POPWI_PLATFORM_ARTEMIS

FN_ATTRIBUTES uint32_t artemis_pop_correlate(int32_t* scoreOut, int32_t guess)
{
	// first argument is not used inside
	return core_pop_correlate((void*)0, DMA2_SAMPLES, signal_comb, COMB_SIZE, scoreOut, guess);
}

FN_ATTRIBUTES uint32_t artemis_pop_data_demodulate(const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert)
{
	// first argument is not used inside
	return core_pop_data_demodulate((void*)0, DMA2_SAMPLES, startSample, dataOut, dataOutSize, invert);
}
#else

uint32_t shannon_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut)
{
	// last argument is not used inside
	return core_pop_correlate(data, dataSize, comb, combSize, scoreOut, 0);
}

uint32_t shannon_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert)
{
	// exactly the same arguments
	return core_pop_data_demodulate(data, dataSize, startSample, dataOut, dataOutSize, invert);
}
#endif


uint32_t counts_per_bits(uint16_t bits)
{
	const double baud = 18181.81818;
	uint16_t counts = (1.0/baud) * ARTEMIS_CLOCK_SPEED_HZ;
	return counts*bits;
}

uint32_t comb_dense_length(void)
{
	return signal_comb[COMB_SIZE-1];
}


// 1296 counts is 27us in 48mhz ticks
#define QUICK_SEARCH_STEPS (1296)



FN_ATTRIBUTES int32_t do_comb(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, uint32_t combOffset)
{
	int16_t j,k;
	uint32_t diff;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	short pol; // signal polarity, comb polarity
	




	xscore = 0; // the "score" of this convolution
	now = start = head = DATA_SAMPLE(0) + combOffset;
	k = 0;
	j = 0;

	nextComb = comb[MIN(k+1, combSize-1)] + start;
	nextSignal = DATA_SAMPLE(j+1);

	// if comb_offset is large enough, we need to skip some edges in the data array, so this scans through edges
	while (now > nextSignal)
	{
		j++;
		nextSignal = DATA_SAMPLE(j+1);
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
			nextSignal = DATA_SAMPLE(j+1);
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


FN_ATTRIBUTES uint32_t core_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut, int32_t guess)
{
	uint32_t denseCombLength = comb[combSize-1] - comb[0];
	uint32_t denseDataLength = 0;

	uint16_t i;

	// we are forced to scan through the input data to determine if any modulus events have occurred in order to get a real value for denseDataLength
	for(i = 1; i < dataSize; i++)
	{
		if( DATA_SAMPLE(i) < DATA_SAMPLE(i-1) )
		{
			denseDataLength += ARTEMIS_CLOCK_SPEED_HZ;
		}

		denseDataLength += DATA_SAMPLE(i)-DATA_SAMPLE(i-1);
	}

	if( denseDataLength < denseCombLength )
	{
		printf("dense data size %"PRIu32" must not be less than dense comb size %"PRIu32"\r\n", denseDataLength, denseCombLength);
		//FIXME: this is not an appropriate way of returning an error condition
		return 0;
	}

	int32_t score, scoreLeft, scoreRight, maxScoreQuick = 0, maxScore = 0; //x(key)score
	uint32_t maxScoreOffsetQuick, maxScoreOffset, scoreOffsetBinSearch, iterations, combOffset;


	// Artemis is given a "guess" of the start timer value when the start-of-frame should occur
#ifdef POPWI_PLATFORM_ARTEMIS
	iterations = guess - DATA_SAMPLE(0) + GUESS_ERROR + 1;
	combOffset = guess - DATA_SAMPLE(0) - GUESS_ERROR;
#else
	iterations = denseDataLength - denseCombLength + 1;
	combOffset = 0;
#endif


	// quick search
	for(; combOffset < iterations; combOffset += QUICK_SEARCH_STEPS)
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

	return DATA_SAMPLE(0) + maxScoreOffset;
}

// pass in a data array including the comb
// pass in the sample which is the end of the comb
FN_ATTRIBUTES uint32_t core_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert)
{
	size_t j,k,jp,kp;
	int32_t xscore; //x(key)score
	uint32_t start, head, now;
	uint32_t nextSignal, nextComb;
	uint8_t dataByte = 0;
	const size_t counts_per_bit = counts_per_bits(1);


	uint32_t combSize = (dataOutSize*8) + 1;


	xscore = 0; // the "score" of this convolution
	now = start = head = startSample;
	kp = k = 0;
	j = 0; // don't set jp, we are about to modify j

	nextComb = counts_per_bits(MIN(k+1, combSize-1)) + start;
	nextSignal = DATA_SAMPLE(j+1);

	// if comb_offset is large enough, we need to skip some edges in the data array, so this scans through edges
	while (now > nextSignal)
	{
		j++;
		nextSignal = DATA_SAMPLE(j+1);
	}

	jp = j;

	while(j < dataSize && k < combSize )
	{
		if(jp&1)
		{
			xscore -= now - head;
		}
		else
		{
			xscore += now - head;
		}

		// if the previous loop set 'now' to a comb edge, we are ready to record a bit
		if( kp != k )
		{
			dataByte <<= 1;
			if( xscore > 0 )
			{
				dataByte |= 1;
			}

			if( k % 8 == 0 )
			{
				if(invert)
				{
					dataByte ^= 0xff;
				}

				dataOut[(k/8)-1] = dataByte;
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
			nextSignal = DATA_SAMPLE(j+1);
		}
		else
		{

			// next event is a comb edge
			k++;
			now = nextComb;

			// prep for next comparison
			nextComb += counts_per_bit;
		}
	} // while
	return 0;
}

unsigned ota_length_encoded(unsigned len)
{
	return len*4;
}

void decode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size)
{
	size_t i,j;
	int bit, bit0, bit1, bit2, bit3;

	*out_size = in_size/4;

	for(i = 0; i < *out_size; i++)
	{
		out[i] = 0x00;
	}

	uint8_t byte_out = 0;
	for(i = 0; i < in_size; i++)
	{
		uint8_t byte_in = in[i];

		for(j = 0; j < 2; j++)
		{
			byte_out <<= 1;
			bit3 = (byte_in & 0x80)?1:-1;
			bit2 = (byte_in & 0x40)?1:-1;
			bit1 = (byte_in & 0x20)?1:-1;
			bit0 = (byte_in & 0x10)?1:-1;

			bit = bit3+bit2+bit1+bit0 > 0;

			if( bit )
			{
				byte_out |= 0x01;
			}

			byte_in <<= 4;
		}

		if( i % 4 == 3 )
		{
			out[i/4] = byte_out;
			byte_out = 0;
		}
	}
}


void encode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size)
{
	size_t i,j;

	int bit;

	*out_size = in_size*4;

	for(i = 0; i < *out_size; i++)
	{
		out[i] = 0x00;
	}

	for(i = 0; i < in_size; i++)
	{
		uint8_t byte = in[i];
		for(j = 0; j < 8; j++)
		{
			bit = (byte & 0x80)?1:0;

			if( bit )
			{
				out[(j/2)+(4*i)] |= 0x0f << ((j%2==0)?4:0);
			}

			byte <<= 1;
		}
	}
}


// helper to set size and checksum before transmitting a packet
void ota_packet_prepare_tx(ota_packet_t* p)
{
	ota_packet_set_size(p);
	ota_packet_set_checksum(p);
}

void ota_packet_set_size(ota_packet_t* p)
{
//	printf("size: %ld  position %d\r\n\r\n", sizeof(*p), (int)offsetof(ota_packet_t,checksum) );
//	printf("data position %d\r\n\r\n", (int)offsetof(ota_packet_t,data) );
//	printf("size: %ld\r\n", sizeof(p->size));
//	printf("size: %ld\r\n", sizeof(p->size));
//	printf("checksum: %ld\r\n", sizeof(p->checksum));
//	printf("type: %ld\r\n", sizeof(p->type));
//	printf("data: %ld\r\n", sizeof(p->data));
//	printf("data.poll: %ld\r\n", sizeof(p->data.poll));
//	printf("data.rpc: %ld\r\n", sizeof(p->data.rpc));
	p->size = strlen(p->data) + (int)offsetof(ota_packet_t,data);
}

// returns actual checksum byte for packet
uint16_t ota_packet_checksum(ota_packet_t* p)
{
	// this assumes that "size" is the first parameter in the struct after checksum
	uint8_t* head = (uint8_t*)p + (int)offsetof(ota_packet_t,size);
	unsigned size = p->size - (int)offsetof(ota_packet_t,size);

	size = MIN(size, sizeof(ota_packet_t) - (int)offsetof(ota_packet_t,size));

	return crcSlow(head, size);
}

// size must be set or else this will not work correctly
void ota_packet_set_checksum(ota_packet_t* p)
{
	uint16_t checksum = ota_packet_checksum(p);
//	printf("checksum: %d\r\n", checksum);
	p->checksum = checksum;
}

// returns non-zero if checksum is ok
short ota_packet_checksum_good(ota_packet_t* p)
{
	uint16_t checksum = ota_packet_checksum(p);
	return (checksum == p->checksum);
}

// fills all bytes in packet with 0
void ota_packet_zero_fill(ota_packet_t* p)
{
	memset(p, 0, sizeof(*p) );
}

// fills all bytes in packet with 0
void ota_packet_zero_fill_data(ota_packet_t* p)
{
	memset(p->data, 0, ARRAY_LEN(p->data) );
}






/*
 * The width of the CRC calculation and result.
 * Modify the typedef for a 16 or 32-bit CRC standard.
 * http://www.barrgroup.com/Embedded-Systems/How-To/CRC-Calculation-C-Code
 */
#define crc_t uint16_t
#define WIDTH  (8 * sizeof(crc_t))
#define TOPBIT (1 << (WIDTH - 1))
#define POLYNOMIAL 0xD8  /* 11011 followed by 0's */

crc_t crcSlow(uint8_t const message[], int nBytes)
{
	crc_t  remainder = 0;
	int byte;
	uint8_t bit;


	/*
	 * Perform modulo-2 division, a byte at a time.
	 */
	for (byte = 0; byte < nBytes; ++byte)
	{
		/*
		 * Bring the next byte into the remainder.
		 */
		remainder ^= (message[byte] << (WIDTH - 8));

		/*
		 * Perform modulo-2 division, a bit at a time.
		 */
		for (bit = 8; bit > 0; --bit)
		{
			/*
			 * Try to divide the current data bit.
			 */
			if (remainder & TOPBIT)
			{
				remainder = (remainder << 1) ^ POLYNOMIAL;
			}
			else
			{
				remainder = (remainder << 1);
			}
		}
	}

	/*
	 * The final remainder is the CRC result.
	 */
	return (remainder);

}   /* crcSlow() */





#ifdef POPWI_PLATFORM_ARTEMIS

uint32_t pop_get_now_slot(void)
{
	return pop_get_slot_pit(pit_get_utc_counts());
}

// how many pit counts until the slot appears
// if we are already in the slot when this is called, we assume it's too late to transmit, so counts till the next available slot is returned
uint32_t pop_get_next_slot_pit(uint32_t slot)
{
	if( slot >= POP_SLOT_COUNT )
	{
		printf("Asking for impossible slot %lu in pop_get_pit_next_slot\r\n", slot);
		return 0;
	}

	uint64_t now = pit_get_utc_counts();
	uint32_t now_slot = pop_get_slot_pit(now);

	if( now_slot >= slot )
	{
		slot += POP_PERIOD_LENGTH;
	}

	uint64_t rounded = now % (POP_SLOT_LENGTH*19200000); // how many pit counts since the beginning of this slot
	uint64_t remaining = ((slot - now_slot) * (POP_SLOT_LENGTH*19200000)) - rounded;

	return remaining;
}

#endif

uint32_t pop_get_slot_pit(uint64_t pit)
{
	uint64_t secs = pit / 19200000; //floor

	return (secs / POP_SLOT_LENGTH) % POP_SLOT_COUNT;
}


uint32_t pop_get_slot_pit_rounded(uint64_t pit)
{
	long double secs = pit / 19200000.0;

	long double fslot = secs / (double) POP_SLOT_LENGTH;

	uint32_t slot = (uint32_t)(fslot + 0.5); // floor

	return slot % POP_SLOT_COUNT;
}









