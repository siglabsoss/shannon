#include "popsymbol.hpp"

#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace pop
{

// when constructing the PopTimestamp, pass in 0.0 for the offset.  This is because the timestamp always applies to this PopSymbol, and therefore the offset coming in with ts is worthless
PopSymbol::PopSymbol(uint8_t s, double m, uint8_t fb, uint8_t c, uint16_t h, PopTimestamp &ts): symbol(s), magnitude(m), fbin(fb), channel(c), host(h), timestamp(ts, 0.0)
{

}

PopSymbol::~PopSymbol()
{

}

void PopSymbol::debug_print()
{
	cout << "  symbol: " << symbol << endl;
	cout << "  magnitude: " << magnitude << endl;
	cout << "  host: " << host << endl;
	cout << "  time: " << boost::lexical_cast<string>(timestamp.get_full_secs()) << "   -   " << boost::lexical_cast<string>(timestamp.get_frac_secs()) << endl;
}

}
