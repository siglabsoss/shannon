
#ifndef __TYPES_CUH_
#define __TYPES_CUH_

#include "poptypes.h"

__device__ double magnitude2( const popComplex& in );

__device__ popComplex operator*(const popComplex& a, const popComplex& b);

__device__ popComplex operator+(const popComplex& a, const popComplex& b);

#endif // __TYPES_CUH_
