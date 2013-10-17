#include "popsymbol.hpp"

namespace pop
{

// when constructing the PopTimestamp, pass in 0.0 for the offset.  This is because the timestamp always applies to this PopSymbol, and therefore the offset coming in with ts is worthless
PopSymbol::PopSymbol(uint8_t s, double m, uint8_t fb, uint8_t c, PopTimestamp &ts): symbol(s), magnitude(m), fbin(fb), channel(c), timestamp(ts, 0.0)
{

}

PopSymbol::~PopSymbol()
{

}


}
