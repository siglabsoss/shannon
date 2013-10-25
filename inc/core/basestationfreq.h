#ifndef __BASESTATION_FREQ_H__
#define __BASESTATION_FREQ_H__


#ifdef __cplusplus
extern "C" {
#endif

extern double bsf_fbin_size();
extern double bsf_fft_bottom_frequency();
extern double bsf_fft_top_frequency();
extern double bsf_fbins_per_channel();
extern double bsf_channel_frequency(unsigned c);
extern double bsf_channel_frequency_above_fft(unsigned c);
extern double bsf_channel_fbin_center(unsigned c);
extern double bsf_channel_fbin_low_exact(unsigned c);
extern double bsf_channel_fbin_high_exact(unsigned c);
extern unsigned bsf_channel_fbin_low(unsigned c);
extern unsigned bsf_channel_fbin_high(unsigned c);


#ifdef __cplusplus
}
#endif


// --- Constants related to channels / frequencies and DSP ---

#define POP_PROTA_BLOCK_A_UPLK 903626953
#define POP_PROTA_BLOCK_B_UPLK 906673828
#define POP_PROTA_BLOCK_C_UPLK 909720703
#define POP_PROTA_BLOCK_D_UPLK 912767578

#define POP_PROTA_BLOCK_A_DOWN 917236328
#define POP_PROTA_BLOCK_B_DOWN 920283203
#define POP_PROTA_BLOCK_C_DOWN 923330078
#define POP_PROTA_BLOCK_D_DOWN 926376953

#define POP_PROTA_BLOCK_A_WIDTH 3200000
#define POP_PROTA_BLOCK_B_WIDTH 3200000
#define POP_PROTA_BLOCK_C_WIDTH 3200000
#define POP_PROTA_BLOCK_D_WIDTH 3200000

#define FFT_SIZE 65536
#define CHAN_SIZE 1040
#define POP_CHANNEL_SPACING ((double) 50781.25)
#define POP_PROTA_BLOCK_A_CHANNEL_0 ((double) 902382812.50)





#endif
