#ifndef __POP_SYMBOL_H_
#define __POP_SYMBOL_H_

#include "mdl/poptimestamp.hpp"
#include <stdint.h>

namespace pop
{

class PopSymbol
{
public:
	PopSymbol(uint8_t, double, uint8_t, uint8_t, PopTimestamp&);
	~PopSymbol();

	uint8_t symbol;
	double magnitude;
	uint8_t fbin;
	uint8_t channel;
	PopTimestamp timestamp;
};



} // namespace pop

#endif
