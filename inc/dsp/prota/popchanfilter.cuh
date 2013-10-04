#ifndef __POP_CHAN_FILTER_CUH__
#define __POP_CHAN_FILTER_CUH__

#include <complex>
#include <cufft.h>

__device__ float magnitude2( cuComplex& in );


#endif