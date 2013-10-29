#ifndef __POP_DECONVOLVE_CUH__
#define __POP_DECONVOLVE_CUH__

extern "C" void gpu_threshold_detection(popComplex* d_in, int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, unsigned peak_sinc_neighbors, int outLenMax, popComplex* h_cts, unsigned *h_maxima_peaks, unsigned *h_maxima_peaks_len, double threshold, int len, int fbins, cudaStream_t* stream);

#endif