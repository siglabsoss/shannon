#ifndef __POP_DECONVOLVE_CUH__
#define __POP_DECONVOLVE_CUH__

#include "dsp/prota/popdeconvolve.hpp"

#ifdef __cplusplus
extern "C" {
#endif


extern void gpu_threshold_detection(popComplex* d_in, int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, unsigned peak_sinc_neighbors, int outLenMax, popComplex* h_cts, unsigned *h_maxima_peaks, unsigned *h_maxima_peaks_len, double threshold, int len, int fbins, cudaStream_t* stream);
extern void gpu_cts_stride_copy(popComplex (*cts_stream_buff)[50][SPREADING_CODES][SPREADING_BINS], popComplex* d_cts, unsigned channel, unsigned spreading_code, unsigned len, unsigned fbins, cudaStream_t* stream);


#ifdef __cplusplus
}
#endif


#endif