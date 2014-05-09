#ifndef __POP_SPARSE_CORRELATE_H__
#define __POP_SPARSE_CORRELATE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pop_correlate(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut);
uint32_t pop_data_demodulate(const uint32_t* data, const uint16_t dataSize, const uint32_t startSample, uint8_t* dataOut, const uint16_t dataOutSize, const short invert);

#ifdef __cplusplus
}
#endif


#endif
