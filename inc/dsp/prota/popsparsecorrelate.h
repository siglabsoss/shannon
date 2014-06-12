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
uint32_t pop_get_next_slot_pit(uint32_t slot);


// both in seconds
#define POP_SLOT_LENGTH (2)
#define POP_PERIOD_LENGTH (60)
#define POP_SLOT_COUNT (POP_PERIOD_LENGTH/POP_SLOT_LENGTH)



#ifdef __cplusplus
}
#endif

#endif
