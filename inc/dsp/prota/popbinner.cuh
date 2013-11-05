#ifndef __POP_BINNER_CUH__
#define __POP_BINNER_CUH__


// the number of samples between expected peaks (FIXME this is a magic number, if sample rate or other stuff changes..)
#define EXPECTED_PEAK_SEPARATION (512+40)

#define EXPECTED_BITS (80)

#define EXPECTED_BITS_TOLERANCE (10)

#define ACCEPTABLE_BITS (EXPECTED_BITS-EXPECTED_BITS_TOLERANCE)


#define MAX_SIGNALS_PER_SPREAD (32) // how much memory to allocate for detecting signal peaks
#define BYTES_PER_DETECTED_PACKET (8) // how much memory to allocate for detecting signal peaks


#ifdef __cplusplus
extern "C" {
#endif


extern void gpu_threshold_detection(const popComplex (*cts_stream_buff)[CHANNELS_USED][SPREADING_CODES][SPREADING_BINS], int* d_out, unsigned int *d_outLen, int* d_maxima_out, unsigned int *d_maxima_outLen, unsigned peak_sinc_neighbors, int outLenMax, popComplex* h_cts, uint8_t(*h_maxima_peaks)[BYTES_PER_DETECTED_PACKET], unsigned *h_maxima_peaks_len, double threshold, int len, int fbins, size_t sample_size, cudaStream_t* stream);

void gpu_threshold_bin(popComplex* d_in);


#ifdef __cplusplus
}
#endif


#endif