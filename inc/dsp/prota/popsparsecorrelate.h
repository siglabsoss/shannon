#ifndef __POP_SPARSE_CORRELATE_H__
#define __POP_SPARSE_CORRELATE_H__

#include <stdint.h>

//#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	uint16_t checksum;
	uint16_t size; // this must be the first param after checksum
	char data[128];
} __attribute__((__packed__)) ota_packet_t;


typedef struct
{
	  uint16_t UIDMH;                                  /**< Unique Identification Register Mid-High, offset: 0x1058 */
	  uint32_t UIDML;                                  /**< Unique Identification Register Mid Low, offset: 0x105C */
	  uint32_t UIDL;                                   /**< Unique Identification Register Low, offset: 0x1060 */
} __attribute__((__packed__)) uuid_parts_t;

typedef struct
{
	union {
		uint8_t bytes[10];
		uuid_parts_t parts;
	} __attribute__((__packed__));
} uuid_t;


extern const uuid_t zero_uuid;


uint32_t artemis_pop_correlate(int32_t* scoreOut, int32_t guess);
uint32_t artemis_pop_data_demodulate(const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

uint32_t shannon_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut);
uint32_t shannon_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);
uint32_t core_pop_llr_demodulate(const uint32_t* data, const size_t dataSize, const uint32_t startSample, int16_t* dataOut, const uint16_t dataOutSize, const short invert);

int32_t do_comb(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, uint32_t combOffset, uint32_t* state);

uint32_t pop_data_demodulate(const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);
void decode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
void encode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
unsigned ota_length_encoded(unsigned len);
uint32_t comb_dense_length(void);
uint16_t crcSlow(uint8_t const message[], int nBytes);
void ota_packet_set_checksum(ota_packet_t* p);
void ota_packet_zero_fill(ota_packet_t* p);
short ota_packet_checksum_good(ota_packet_t* p);
void ota_packet_prepare_tx(ota_packet_t* p);
void ota_packet_set_size(ota_packet_t* p);
uint32_t counts_per_bits(uint16_t bits);
void ota_packet_zero_fill_data(ota_packet_t* p);
uint32_t pop_get_now_slot(void);
uint64_t pop_get_next_slot_pit(uint32_t slot);
uint32_t pop_get_slot_pit(uint64_t pit);
uint32_t pop_get_slot_pit_rounded(uint64_t pit);
int64_t pop_get_slot_error(uint32_t, uint64_t);
double pop_get_slot_pit_float(uint64_t pit);


// both in seconds
#define POP_SLOT_LENGTH (2)
#define POP_PERIOD_LENGTH (60)
#define POP_SLOT_COUNT (POP_PERIOD_LENGTH/POP_SLOT_LENGTH)
// chosen so that if P(0) = 0.00001 and P(1) = 0.99999 the LLR will still have some headroom on an int16_t
#define LLR_SCALE (3600)

#define BAUD_RATE (9090.90909090909000)
#define COUNTS_PER_BIT ((uint32_t)(48e6/BAUD_RATE))
#define CALC_LLR_P0(xscore) (((xscore)+COUNTS_PER_BIT)/((double)COUNTS_PER_BIT*2))  // chance that bit is a 0



#ifdef __cplusplus
}
#endif

#endif
