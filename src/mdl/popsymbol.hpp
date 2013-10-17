#ifndef __POP_SYMBOL_H_
#define __POP_SYMBOL_H_

#include "mdl/poptimestamp.hpp"

namespace pop
{

class PopSymbol
{
public:
	PopSymbol();
	~PopSymbol();

	unsigned int symbol;
	PopTimestamp timestamp;
};



} // namespace pop

#endif
