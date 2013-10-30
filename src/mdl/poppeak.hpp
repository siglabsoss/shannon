#ifndef __POP_PEAK_H_
#define __POP_PEAK_H_

#include "mdl/poptimestamp.hpp"
#include <dsp/common/poptypes.h>
#include <stdint.h>


#define PEAK_SINC_NEIGHBORS (7)     // how many samples to add to either side of a local maxima for sinc interpolate
#define PEAK_SINC_SAMPLES_TOTAL (PEAK_SINC_NEIGHBORS+PEAK_SINC_NEIGHBORS+1) // how many total samples are needed for sinc interpolation
#define PEAK_SINC_SAMPLES (100000)  // how many samples to sinc interpolate around detected peaks








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

	// the upper left fbin represented by this class (fbin+1 is the detected peak's fbin).  this value is the smallest fbin represented by this peak
	uint8_t fbin;

	// left most sample's x which is a rolling number.  this value is the smallest sample_x represented by this peak
	size_t sample_x;

	// channel of this peak
	uint8_t channel;

	// basestation's id
	uint16_t basestation;

	data_point_t data[PEAK_SINC_SAMPLES_TOTAL*3];

//	void debug_print();
};



} // namespace pop

#endif
