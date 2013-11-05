#ifndef __POP_BINNER_CUH__
#define __POP_BINNER_CUH__


// the number of samples between expected peaks (FIXME this is a magic number, if sample rate or other stuff changes..)
#define EXPECTED_PEAK_SEPARATION (512+40)

#define EXPECTED_BITS (80)

#define EXPECTED_BITS_TOLERANCE (10)

#define ACCEPTABLE_BITS (EXPECTED_BITS-EXPECTED_BITS_TOLERANCE)


#ifdef __cplusplus
extern "C" {
#endif


extern void gpu_threshold_detection(const popComplex (*cts_stream_buff)[50][SPREADING_CODES][SPREADING_BINS], int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, unsigned peak_sinc_neighbors, int outLenMax, popComplex* h_cts, unsigned *h_maxima_peaks, unsigned *h_maxima_peaks_len, double threshold, int len, int fbins, size_t sample_size, cudaStream_t* stream);
//extern void gpu_cts_stride_copy(popComplex (*cts_stream_buff)[50][SPREADING_CODES][SPREADING_BINS], popComplex* d_cts, unsigned channel, unsigned spreading_code, unsigned len, unsigned fbins, cudaStream_t* stream);

void gpu_threshold_bin(popComplex* d_in);


#ifdef __cplusplus
}
#endif


#endif