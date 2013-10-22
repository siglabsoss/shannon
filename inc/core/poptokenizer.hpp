#ifndef __POP_TOKENIZER__
#define __POP_TOKENIZER__

/******************************************************************************
 * Copyright 2013 PopWi Technology Group, Inc. (PTG)
 *
 * This file is proprietary and exclusively owned by PTG or its associates.
 * This document is protected by international and domestic patents where
 * applicable. All rights reserved.
 *
 ******************************************************************************/

//#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "mdl/popsymbol.hpp"
//#include "popradio.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
//#include "json/json.h"
//#include "core/utilities.hpp"


//using boost::asio::ip::udp;
//using namespace boost::asio;
using namespace std;



namespace pop
{

class PopTokenizer : public PopSink<PopSymbol>, public PopSource<PopSymbol>
{
public:

	vector<PopSymbol> symbols;

	PopTokenizer() : PopSink<PopSymbol>( "PopTokenizer" ), PopSource<PopSymbol>( "PopTokenizer" )
	{}

	void init()
	{

	}



	void process(const PopSymbol* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
	{
		for( size_t i = 0; i < data_size; i++ )
		{
			symbols.push_back(data[i]);
		}

		cout << "holding " << symbols.size() << " symbols" << endl;

		if( symbols.size() > 10 )
			tokenize();

	}

	void tokenize()
	{

		// sort Symbols in order
		std::sort(symbols.begin(), symbols.end(), PopSymbol::timestamp_comparitor);

		for( unsigned i = 0; i < symbols.size() - 1; i++ )
		{
			PopSymbol *current = &symbols[i];
			PopSymbol *next = &symbols[i+1];

			if( abs(current->timestamp.get_real_secs() - next->timestamp.get_real_secs()) < 0.001 )
			{
				PopTimestamp difference;// = PopTimestamp(current);

				if( current->timestamp.get_real_secs() < next->timestamp.get_real_secs() )
				{
					difference = PopTimestamp(next->timestamp);
					difference -= current->timestamp;

					cout << "these two stamps (" << i << ", " << i+1 << ") have a difference of " << difference.get_frac_secs() << endl;
					current->debug_print();
					next->debug_print();

				}



			}
			else
			{
//				cout << "these two stamps (" << i << ", " << i+1 << ") are too far apart: " << endl;
//				current->debug_print();
//				next->debug_print();
//
//				cout << endl << endl;
			}

//			PopTimestamp difference = PopTimestamp

		}

		// delete all but the last symbol
		int erase = symbols.size() - 1;

		// delete vector elements
		for( int i = 0; i < erase; i++ )
			symbols.erase(symbols.begin());
	}




};
} // namespace pop


#endif
