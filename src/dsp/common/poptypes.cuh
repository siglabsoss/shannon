
#ifndef __TYPES_CUH_
#define __TYPES_CUH_

#include "poptypes.h"

__host__ __device__ double magnitude2( const popComplex& in );

__host__ __device__ popComplex operator*(const popComplex& a, const popComplex& b);

__host__ __device__ popComplex operator+(const popComplex& a, const popComplex& b);

#endif // __TYPES_CUH_
