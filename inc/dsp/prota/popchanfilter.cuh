#ifndef __POP_CHAN_FILTER_CUH__
#define __POP_CHAN_FILTER_CUH__

#include <complex>
#include <cufft.h>
#include <dsp/common/poptypes.cuh>


#ifdef __cplusplus
extern "C" {
#endif


extern cudaStream_t chan_filter_stream;


#ifdef __cplusplus
}
#endif


#endif