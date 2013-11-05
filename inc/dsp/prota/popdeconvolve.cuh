#ifndef __POP_DECONVOLVE_CUH__
#define __POP_DECONVOLVE_CUH__

#include "dsp/prota/popdeconvolve.hpp"

#ifdef __cplusplus
extern "C" {
#endif

extern void gpu_cts_stride_copy(popComplex (*cts_stream_buff)[50][SPREADING_CODES][SPREADING_BINS], popComplex* d_cts, unsigned channel, unsigned spreading_code, unsigned len, unsigned fbins, cudaStream_t* stream);


#ifdef __cplusplus
}
#endif


#endif