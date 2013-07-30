#include <memory>
#include <iostream>	

#include <math.h>

#include "dsp/popgenerate.hpp"
#include "net/popnetwork.hpp"

/******************************************************************************
 *
 * POPWI PFROTOCOL A (UPLINK)
 *
 * Revision 0.1
 *
 *****************************************************************************/

#define POP_SAMPLE_RATE 3200000
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

#define POP_GAUSSIAN_BT 0.5 // gaussian BT coefficient
#define POP_GAUSSIAN_NT 2 // number of symbol periods between beginning and peak of Gaussian impulse response


namespace pop
{
	SYMBOL_SET code_a;
	SYMBOL_SET code_b;
	SYMBOL_SET code_c;

	PN_CODE GMSK_code;


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
	 *	Generates a gaussian filtered bitstream from a given PN sequence. 
	 *	> Configured for a two-symbol modulator, such as 2-GMSK
	 *  > Outputs a vector of floats, normalized to [0 1]
	 *  >> Must supply pointer for output data - size = sizeof(float)*pn_len*oversamp_factor
	 */
	void popGenGaussian(const uint8_t * codeIn, float * out, uint32_t pn_len, uint32_t oversamp_factor){
		uint32_t gaussian_size, os_idx;
		uint32_t gauss_idx, oversamp_size, i, j; 
		uint32_t symbol, byteIdx, byteMod;
		uint32_t os_byteIdx, os_byteMod; 
		double gaussian_sum;

		double tSymbol = 1.0/POP_CODE_A_CHIP_RATE;
		double gauss_b = POP_GAUSSIAN_BT/tSymbol;
		double gauss_a = 1.0/gauss_b * sqrt(log(2)/2);

		printf("gauss params: %f : %f : %f \n", tSymbol, gauss_b, gauss_a);

		// Allocate arrays
		gaussian_size = (POP_GAUSSIAN_NT * oversamp_factor * 2) + 1;
		oversamp_size = pn_len * oversamp_factor;

		double arr_t [gaussian_size];
		double gaussian [gaussian_size];
		uint8_t oversampled_code [oversamp_size]; // init to 0

		double t_start = -POP_GAUSSIAN_NT*tSymbol;
		double dt = tSymbol/oversamp_factor;

		// Populate arrays
		for(i = 0; i < gaussian_size; i++){
			arr_t[i] = t_start + (dt * i);
			gaussian[i] = sqrt(M_PI)/gauss_a * exp(-pow(M_PI * arr_t[i] / gauss_a, 2));
			gaussian_sum += gaussian[i];
		}

		// Init
		for(i = 0; i < oversamp_size; i++){
			oversampled_code[i] = 0;
		}

		// Normalize Gaussian
		for(i = 0; i < gaussian_size; i++){
			gaussian[i] /= gaussian_sum;
		}

		// Generate oversampled (original) data vector
		for(i = 0; i < pn_len; i++){
			byteIdx = i / 8;
			byteMod = i % 8;
			symbol = (codeIn[byteIdx] >> (7-byteMod)) & 0x1;
			//printf("byte: %X, ")
			for(j = 0; j < oversamp_factor; j++){
				os_idx = i * oversamp_factor + j;
				os_byteIdx = os_idx / 8;
				os_byteMod = os_idx % 8;
				oversampled_code[os_byteIdx] |= (symbol << (7-os_byteMod));
			}
		}

		// Apply Gaussian filter to oversampled data vector
		// Todo: verify PN edge filter truncation
		for(i = 0; i < oversamp_size; i++){
			// ensure starting with empty mem
			out[i] = 0;

			for(j = 0; j < gaussian_size; j++){
				// condition index of gaussian step response
				gauss_idx = i - (floor(gaussian_size/2)) + j;
				if(gauss_idx < 0) gauss_idx = 0; // truncate early overrun
				if(gauss_idx >= oversamp_size) gauss_idx = oversamp_size - 1; // truncate late overrun

				// Find the symbol
				byteIdx = gauss_idx / 8;
				byteMod = gauss_idx % 8;
				symbol = (oversampled_code[byteIdx] >> (7-byteMod)) & 0x1;

				// Compute
				out[i] += (float)symbol * (float)gaussian[j];
			}
		}

		// DEBUG - CHECK VECTORS
		if(1){
			int nDebugBytes = 4;
			printf(" ---- popGenGaussian DEBUG --- \n");

			printf("PN CODE: ");
			for( i = 0; i < nDebugBytes; i++){
				printf("0x%X ",codeIn[i]);
			}
			printf("\n");

			printf("RAW: ");
			for( i = 0; i < nDebugBytes*8; i++){
				byteIdx = i / 8;
				byteMod = i % 8;
				symbol = (codeIn[byteIdx] >> (7-byteMod)) & 0x1;
				printf("%d",symbol);
			}
			printf("\n");

			printf("OVERSAMP: ");
			for( i = 0; i < nDebugBytes*8; i++){
				byteIdx = i / 8;
				byteMod = i % 8;
				symbol = (oversampled_code[byteIdx] >> (7-byteMod)) & 0x1;
				printf("%d",symbol);
			}
			printf("\n");
			printf(" ---- /popGenGaussian DEBUG --- \n");
		}

		// returns gaussian-smoothed PN code, with oversampling, to float* out
	}


	void popGenGMSK(const uint8_t *codeIn, std::complex<float> *out, int pn_len, int oversamp_factor){
		uint32_t oversamp_pn_len, idx;
		double symbol, ref_chan;
		float ref_freq, ref_freq_dev;

		oversamp_pn_len = pn_len * oversamp_factor;

		// Allocate arrays
		GMSK_code.resize(oversamp_pn_len);
		float filteredCode [oversamp_pn_len];

		// Apply gaussian to PN code
		popGenGaussian(codeIn, filteredCode, pn_len, oversamp_factor);
		// DEBUG - CHECK VECTORS
		if(1){
			int nDebugBytes = 4;
			printf(" ---- popGenGMSK DEBUG --- \n");

			printf("GAUSS DATA: ");
			for(int i = 0; i < nDebugBytes*8; i++){
				printf("%f ",filteredCode[i]);
			}
			printf("\n");

			printf(" ---- /popGenGMSK DEBUG --- \n");
		}
		// Modulate the code
		for( idx = 0; idx < oversamp_pn_len; idx++ )
		{
			// Get the symbol (this is a double!)
			symbol = filteredCode[idx];

			/// MSK (a specific case of FSK) frequency deviation
			/// TODO: make adjustable for frequency sample points
			ref_freq_dev = (float)symbol * (float)POP_CODE_B_FREQ_DEV;

			/// modulation frequency
			/// TODO: make adjustable for frequency sample points
			ref_chan = 0;

			ref_freq = (float)ref_chan * (float)POP_CODE_C_CHANNEL_SPACING +
			           (float)POP_CODE_B_BASEBAND + 0.0 + ref_freq_dev;

			GMSK_code[idx].real(sinf(2.0*M_PI*ref_freq*(float)idx / (float(oversamp_factor)) ));
			GMSK_code[idx].imag(cosf(2.0*M_PI*ref_freq*(float)idx / (float(oversamp_factor)) ));
		}

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
		//popAlgo001();
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
