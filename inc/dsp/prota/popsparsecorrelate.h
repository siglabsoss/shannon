#ifndef __POP_SPARSE_CORRELATE_H__
#define __POP_SPARSE_CORRELATE_H__

#include <stdint.h>

//#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t artemis_pop_correlate(int32_t* scoreOut, int32_t guess);
uint32_t artemis_pop_data_demodulate(const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

uint32_t shannon_pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut);
uint32_t shannon_pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

uint32_t pop_data_demodulate(const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);
void decode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
void encode_ota_bytes(uint8_t* in, uint32_t in_size, uint8_t* out, uint32_t* out_size);
unsigned ota_length_encoded(unsigned len);
uint32_t comb_dense_length(void);

#ifdef __cplusplus
}
#endif

#endif
