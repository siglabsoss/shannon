
#ifndef __TYPES_CUH_
#define __TYPES_CUH_

#include "poptypes.h"
#include <stdint.h>

__host__ __device__ double magnitude2( const popComplex& in );

__host__ __device__ popComplex operator*(const popComplex& a, const popComplex& b);

__host__ __device__ popComplex operator+(const popComplex& a, const popComplex& b);

__host__ __device__ void pak_change_bit(uint8_t storage[], unsigned index, unsigned value);

__host__ void pak_print(uint8_t storage[], unsigned count);

#endif // __TYPES_CUH_
