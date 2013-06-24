#include "popgenerate.hpp"
#include "popnetwork.hpp"
#include <memory>
#include <iostream>	

#include <math.h>

/******************************************************************************
 *
 * POPWI PFROTOCOL A (UPLINK)
 *
 * Revision 0.1
 *
 *****************************************************************************/

#define POP_SAMPLE_RATE 1000000
#define POP_FREQ_SAMPLE_POINTS 32 // number of frequency sample points
#define POP_FREQ_ERROR 160 // frequency sample range in part-per-million

#define POP_CODE_A_SYMBOLS 265 // number of symbols per code
#define POP_CODE_A_CHIPS 32767 //number of chips per symbol
#define POP_CODE_A_CHIP_RATE 25986 //chips per second
#define POP_CODE_A_CHIPS_PER_HOP 512 //hz
#define POP_CODE_A_FREQ_DEV 12993 //hz (difference between mark and space)
#define POP_CODE_A_HOPPING_PERIOD 20000 //ns
#define POP_CODE_A_CHANNEL_SPACING 25391 //hz
#define POP_CODE_A_CHANNELS 50 //number of spreading channels
#define POP_CODE_A_BASEBAND 0 //hz

#define POP_CODE_B_SYMBOLS 1 // 265 // number of symbols per code
#define POP_CODE_B_CHIPS 8192 //number of chips per symbol
#define POP_CODE_B_CHIP_RATE 25986 //chips per second
#define POP_CODE_B_CHIPS_PER_HOP 512 //hz
#define POP_CODE_B_FREQ_DEV 12993 //hz (difference between mark and space)
#define POP_CODE_B_HOPPING_PERIOD 20000 //ns
#define POP_CODE_B_CHANNEL_SPACING 25391 //hz
#define POP_CODE_B_CHANNELS 50 //number of spreading channels
#define POP_CODE_B_BASEBAND 0 //hz

#define POP_CODE_C_SYMBOLS 265 // number of symbols per code
#define POP_CODE_C_CHIPS 512 //number of chips per symbol
#define POP_CODE_C_CHIP_RATE 25986 //chips per second
#define POP_CODE_C_CHIPS_PER_HOP 512 //hz
#define POP_CODE_C_FREQ_DEV 12993 //hz (difference between mark and space)
#define POP_CODE_C_HOPPING_PERIOD 20000 //ns
#define POP_CODE_C_CHANNEL_SPACING 25391 //hz
#define POP_CODE_C_CHANNELS 50 //number of spreading channels
#define POP_CODE_C_BASEBAND 0 //hz

namespace pop
{
	SYMBOL_SET code_a;
	SYMBOL_SET code_b;
	SYMBOL_SET code_c;


	// frequency hopping channel sequence
	uint8_t __pop_hop_sequence[] = { 9, 44, 45, 17, 30, 18, 3, 48, 11, 13, 35,
	                                 22, 14, 15, 49, 8, 26, 6, 29, 37, 32, 16,
	                                 46, 33, 24, 42, 47, 0, 5, 28, 19, 10, 4,
	                                 39, 40, 43, 25, 31, 1, 7, 23, 27, 12, 41,
	                                 34, 21, 2, 36, 38, 20 };
	
	uint32_t __idx1 = 0;
	std::complex<float> __temp_return[1500];

	void *popCallback(void* data,std::size_t size)
	{
		memcpy((void*)__temp_return, (void*)&code_b[0][0][0], 12000);

		return __temp_return;
	}

	/**
	 * Uses a dot product operation to generate new pseudorandom codes.
	 * @param base Input to vector operation.
	 * @param base_start Pointer to beginning of base used for
	 * circular buffering.
	 * @param hash Input to vector operation.
	 * @param has_start Pointer to beginning of has used for circular
	 * buffering.
	 * @param output Output buffer.
	 * @param output_start Pointer to beginning of output used for circular
	 * buffering.
	 * @param len Length of base, hash and output buffers.
	 */
	void popAlgo001GenCodes(uint8_t base[], uint32_t base_start,
		                    uint8_t hash[], uint32_t hash_start,
		                    uint8_t output[], uint32_t output_start,
		                    uint32_t len)
	{
		uint32_t m, n, base_idx, hash_idx, output_idx;

		for( m = 0; m < len; m++ )
		{
			output_idx = (m + output_start) % len;
			hash_idx = (m + hash_start) % len;

			output[output_idx] = 0;

			for( n = 0; n < len; n++ )
			{
				base_idx = (n + base_start) % len;
				output[output_idx] += base[base_idx] * hash[hash_idx];
			}
		}
	}

	void popAlgo001()
	{
		uint32_t m, n, samp_len, ref_idx, ref_symbol_idx, ref_symbol_byte_idx;
		uint32_t ref_samp_per_symbol, ref_chan, ref_sequence;
		uint8_t ref_symbol, ref_symbol_byte_mod;
		float ref_freq, ref_freq_dev;

		printf("[popwi / popgenerate] --------------------------------------\r\n");
		printf("[popwi / popgenerate] generating PopWi Protocol A Codes\r\n");
		printf("[popwi / popgenerate] --------------------------------------\r\n");
		printf("[popwi / popgenerate]    CodeB:\r\n");
		printf("[popwi / popgenerate]       generating %d symbols\r\n", POP_CODE_B_SYMBOLS);
		printf("[popwi / popgenerate]       generating %d frequency sample points\r\n", POP_FREQ_SAMPLE_POINTS);
		printf("[popwi / popgenerate]       generating %d chips per symbol\r\n", POP_CODE_B_CHIPS);
		printf("[popwi / popgenerate]          chips per second (nominal): %d\r\n", POP_CODE_B_CHIP_RATE);


		code_b.resize(POP_CODE_B_SYMBOLS);

		for( m = 0; m < POP_CODE_B_SYMBOLS; m++ )
		{
			code_b[m].resize(POP_FREQ_SAMPLE_POINTS);
			for( n = 0; n < POP_FREQ_SAMPLE_POINTS; n++ )
			{
				/// number of samples per waveform
				samp_len = (uint32_t)(((uint64_t)POP_CODE_B_CHIPS * (uint64_t)POP_SAMPLE_RATE) / (uint64_t)POP_CODE_B_CHIP_RATE);

				printf("[popwi / popgenerate]          symbol #%d, f.s. #%d, samples: %d\r\n", m, n, samp_len);

				code_b[m][n].resize(samp_len);

				// loop over sample window and generate reference waveform
				for( ref_idx = 0; ref_idx < samp_len; ref_idx++ )
				{
					/// pay careful attention to this... Using integer math to
					/// decimate the sample count into the symbol index!
					ref_symbol_idx = (POP_CODE_B_CHIPS * ref_idx) / samp_len;

					/// byte which contains the current symbol
					ref_symbol_byte_idx = ref_symbol_idx / 8; // bits per byte

					/// symbol position in byte
					ref_symbol_byte_mod = ref_symbol_idx % 8;

					/// reference symbol should be mark(1) or space(0)
					ref_symbol = (__code_m4k_001[ref_symbol_byte_idx] >> ref_symbol_byte_mod) & 0x1;

					/// frequency hoping sequence
					/// TODO: implement frequency hoping
					ref_sequence = 0;

					/// frequency hoping channel
					ref_chan = __pop_hop_sequence[ref_sequence];

					/// MSK (a specific case of FSK) frequency deviation
					/// TODO: make adjustable for frequency sample points
					ref_freq_dev = (float)ref_symbol * (float)POP_CODE_B_FREQ_DEV;

					/// modulation frequency
					/// TODO: make adjustable for frequency sample points
					ref_freq = (float)ref_chan * (float)POP_CODE_C_CHANNEL_SPACING +
					           (float)POP_CODE_B_BASEBAND + 0.0 + ref_freq_dev;

					code_b[m][n][ref_idx].real(sinf(2.0*M_PI*ref_freq*(float)ref_idx / (float(samp_len)) ));
					code_b[m][n][ref_idx].imag(cosf(2.0*M_PI*ref_freq*(float)ref_idx / (float(samp_len)) ));
				}
			}
		}

	}

	void popGenerateConstants()
	{
		printf("\r\n");
		printf("[popwi / popgenerate] nominal sample rate: %d\r\n", POP_SAMPLE_RATE);
		popAlgo001();
		printf("\r\n");
	}

	void popGenerateInit()
	{
		popGenerateConstants();
	}

	void popGenerateDeinit()
	{

	}

}
