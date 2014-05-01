#ifndef __POP_SPARSE_CORRELATE_H__
#define __POP_SPARSE_CORRELATE_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//int b64_encode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );
//int b64_decode( const char *inbytes, unsigned count, char *outbytes, unsigned *countOut );
//unsigned b64_length_encoded(unsigned len);

int32_t pop_correlate(uint32_t* data, uint16_t dataSize, uint32_t* comb, uint32_t combSize);


#ifdef __cplusplus
}
#endif


#endif
