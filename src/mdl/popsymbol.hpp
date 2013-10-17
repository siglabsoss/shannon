#ifndef __POP_SYMBOL_H_
#define __POP_SYMBOL_H_

#include "mdl/poptimestamp.hpp"
#include <stdint.h>

namespace pop
{

class PopSymbol
{
public:
	PopSymbol();
	~PopSymbol();

	uint8_t symbol;
	double magnitude;
	uint8_t fbin;
	uint8_t channel;
	PopTimestamp timestamp;
};



} // namespace pop

#endif
