#ifndef __POP_SYMBOL_H_
#define __POP_SYMBOL_H_

#include "mdl/poptimestamp.hpp"
#include <stdint.h>

namespace pop
{

class PopSymbol
{
public:
	PopSymbol(uint8_t, double, uint8_t, uint8_t, uint16_t, PopTimestamp&);
	~PopSymbol();

	uint8_t symbol;
	double magnitude;
	uint8_t fbin;
	uint8_t channel;
	uint16_t host;
	PopTimestamp timestamp;

	static bool timestamp_comparitor (const PopSymbol &a, const PopSymbol &b) { return (a.timestamp.get_real_secs()<b.timestamp.get_real_secs()); }

	void debug_print();
};



} // namespace pop

#endif
