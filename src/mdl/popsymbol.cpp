#include "popsymbol.hpp"

#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace std;

namespace pop
{

PopSymbol::PopSymbol(uint8_t s, double m, uint8_t fb, uint8_t c, uint16_t h, PopTimestamp &ts): symbol(s), magnitude(m), fbin(fb), channel(c), host(h), timestamp(ts)
{

}

PopSymbol::~PopSymbol()
{

}

void PopSymbol::debug_print()
{
	cout << "  symbol: " << (int)symbol << endl;
	cout << "  magnitude: " << magnitude << endl;
	cout << "  host: " << host << endl;
	cout << "  time: " << boost::lexical_cast<string>(timestamp.get_full_secs()) << "   -   " << boost::lexical_cast<string>(timestamp.get_frac_secs()) << endl;
}

}
