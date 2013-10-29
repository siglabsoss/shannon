#ifndef __POP_PEAK_H_
#define __POP_PEAK_H_


#include "dsp/prota/popdeconvolve.hpp"
#include "mdl/poptimestamp.hpp"

#include <stdint.h>




namespace pop
{

typedef struct __attribute__ ((__packed__))
{
	PopTimestamp timestamp;
	popComplex sample;
} data_point_t;

class  __attribute__ ((__packed__)) PopPeak
{
public:
//	PopSymbol(uint8_t, double, uint8_t, uint8_t, uint16_t, PopTimestamp&);
//	~PopSymbol();

	// the symbol of this detected peak
	uint8_t symbol;

	// the upper left fbin represented by this class (fbin+1 is the detected peak's fbin)
	uint8_t fbin;

	// left most sample's x which is a rolling number
	uint16_t sample_x;

	// channel of this peak
	uint8_t channel;

	// basestation's id
	uint16_t basestaion;

	data_point_t data[PEAK_SINC_SAMPLES_TOTAL*3];

//	void debug_print();
};



} // namespace pop

#endif
