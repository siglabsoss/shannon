#ifndef __POP_SPARSE_CORRELATE_H__
#define __POP_SPARSE_CORRELATE_H__

#include <stdint.h>

//#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum
{
	 OTA_PACKET_POLL = 0  // Polling for location and also possibly rx packets
	,OTA_PACKET_RPC = 10
} OTA_PACKET_TYPE_T;


typedef struct {
	uint16_t unused;
} __attribute__((__packed__)) ota_packet_poll_data_t;

typedef struct {
	char name[32];
	int32_t p0;
	int32_t p1;
	int32_t p2;
} __attribute__((__packed__)) ota_packet_rpc_data_t;

// just here wasting memory
typedef struct {
	char bloat[1024];
} __attribute__((__packed__)) ota_packet_bloat_data_t;

typedef struct
{
	uint8_t checksum;
	uint16_t size; // this must be the first param after checksum
	uint8_t type; // use OTA_PACKET_TYPE_T when setting this
	union __attribute__((__packed__)) {
		ota_packet_poll_data_t  poll;
		ota_packet_rpc_data_t   rpc;
		ota_packet_bloat_data_t bloat;
	} data;
} __attribute__((__packed__)) ota_packet_t;




uint32_t artemis_pop_correlate(int32_t* scoreOut, int32_t guess);
uint32_t artemis_pop_data_demodulate(const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

uint32_t shannon_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut);
uint32_t shannon_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

uint32_t pop_data_demodulate(const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);
void decode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
void encode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
unsigned ota_length_encoded(unsigned len);
uint32_t comb_dense_length(void);
uint8_t crcSlow(uint8_t const message[], int nBytes);
void ota_packet_set_checksum(ota_packet_t* p);
void ota_packet_zero_fill(ota_packet_t* p);
short ota_packet_checksum_good(ota_packet_t* p);
void ota_packet_prepare_tx(ota_packet_t* p);
void ota_packet_set_size(ota_packet_t* p);




#ifdef __cplusplus
}
#endif

#endif
