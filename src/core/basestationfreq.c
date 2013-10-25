


#ifndef __BASESTATION_FREQ_C__
#define __BASESTATION_FREQ_C__

#include "core/basestationfreq.h"
#include <math.h>

double bsf_fbin_size()
{
	return 1.0 / ((double)FFT_SIZE / POP_PROTA_BLOCK_A_WIDTH);
}

double bsf_fft_bottom_frequency()
{
	return POP_PROTA_BLOCK_A_UPLK - (double) POP_PROTA_BLOCK_A_WIDTH / 2;
}

double bsf_fft_top_frequency()
{
	return POP_PROTA_BLOCK_A_UPLK + (double) POP_PROTA_BLOCK_A_WIDTH / 2;
}

double bsf_fbins_per_channel()
{
	return POP_CHANNEL_SPACING / bsf_fbin_size();
}

double bsf_channel_frequency(unsigned c)
{
	return POP_PROTA_BLOCK_A_CHANNEL_0 + c * POP_CHANNEL_SPACING;
}

double bsf_channel_frequency_above_fft(unsigned c)
{
	return bsf_channel_frequency(c) - bsf_fft_bottom_frequency();
}

double bsf_channel_fbin_center(unsigned c)
{
	return bsf_channel_frequency_above_fft(c) / bsf_fbin_size();
}

double bsf_channel_fbin_low_exact(unsigned c)
{
	return bsf_channel_fbin_center(c) - bsf_fbins_per_channel() / 2;
}

double bsf_channel_fbin_high_exact(unsigned c)
{
	return bsf_channel_fbin_center(c) + bsf_fbins_per_channel() / 2;
}

// rounded versions of the above
unsigned bsf_channel_fbin_low(unsigned c)
{
	return round( bsf_channel_fbin_low_exact(c) );
}

unsigned bsf_channel_fbin_high(unsigned c)
{
	return round( bsf_channel_fbin_high_exact(c) );
}

unsigned bsf_zero_shift_channel_fbin_low(unsigned c)
{
	return (bsf_channel_fbin_low(c) + (FFT_SIZE/2)) % FFT_SIZE;
}

unsigned bsf_zero_shift_channel_fbin_high(unsigned c)
{
	return (bsf_channel_fbin_high(c) + (FFT_SIZE/2)) % FFT_SIZE;
}

#endif
