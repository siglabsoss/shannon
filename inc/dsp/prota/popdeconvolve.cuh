#ifndef __POP_DECONVOLVE_CUH__
#define __POP_DECONVOLVE_CUH__

#include "dsp/prota/popdeconvolve.hpp"
#include "core/basestationfreq.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void gpu_cts_stride_copy(double (*cts_stream_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], popComplex* d_cts, unsigned channel, unsigned spreading_code, unsigned len, unsigned fbins, cudaStream_t* stream);


#ifdef __cplusplus
}
#endif


#endif