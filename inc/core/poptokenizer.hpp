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
#include "core/phymessage.h"
#include "mdl/poppeak.hpp"
#include "mdl/popsymbol.hpp"
//#include "popradio.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
//#include "core/utilities.hpp"


using namespace boost;
using namespace std;

// cycles divided by clock rate gives us symbol separation in seconds
#define _SYMBOL_SEP_BASE ((double) 282624 / 26000000)

// add in 150ppm tolerance on the crystal
#define _SYMBOL_SEP_ERROR ( _SYMBOL_SEP_BASE * 1.000150 )


#define EXPECTED_BIN_SEPARATION (100)

// width of each bin in seconds. smaller bins allow for more timeslices
#define TOKENIZER_BIN_LENGTH ( _SYMBOL_SEP_ERROR / EXPECTED_BIN_SEPARATION ) // 0.00010871784369230769

// number of bins per second, rounded up
#define BIN_COUNT (  (unsigned int) ceil( 1.0 / TOKENIZER_BIN_LENGTH ) )     // 9199

// number of base stations we support
#define MAX_BASE_STATIONS (2)

// delay in seconds which allows all packets to arrive in any order
#define PIPELINE_DELAY (1)

// length of pipeline in memory in seconds
#define PIPELINE_LENGTH (4)

#define PIPELINE_BINS (BIN_COUNT * PIPELINE_LENGTH)

namespace pop
{

class PopTokenizer : public PopSink<PopPeak>, public PopSource<PopPeak>
{
public:

	vector<boost::tuple<unsigned short, const PopPeak*> > symbols[MAX_BASE_STATIONS];
	PopTimestamp start_time;
	PopTimestamp newest_time;
	size_t last_comb;

	PopTokenizer() : PopSink<PopPeak>( "PopTokenizer" ), PopSource<PopPeak>( "PopTokenizer" ), last_comb(0)
	{
		// FIXME: grab from real clock
		newest_time = start_time = PopTimestamp(1383178239.0 + 3);

		// resize the pipeline of each basestation
		for( int i = 0; i < MAX_BASE_STATIONS; i++ )
		{
			symbols[i].resize(PIPELINE_BINS);
		}

		cout << "bin count " << BIN_COUNT << endl;
		cout << "PHY_MSG " << sizeof(PHY_MSG) << endl;
		cout << "PHY_MSG_HELO " << sizeof(PHY_MSG_HELO) << endl;

	}

	void init()
	{
	}



	void process(const PopPeak* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{

		static PopTimestamp ts;
		static bool set = false;
		static size_t count = 0;

//		cout << "count " << count << endl << endl;

		count ++;

		for( size_t i = 0; i < data_size; i++ )
		{
			// create reference for easy syntax below
			const PopPeak &peak = data[0];

			// debug skip
//			if( peak.basestation == 0 )
//				continue;

			// remember that the actual timestamp we want is the center sample
			const PopTimestamp &peak_timestamp = peak.data[PEAK_SINC_NEIGHBORS+1].timestamp;

			// create mutable copy
			PopTimestamp difference = PopTimestamp(peak_timestamp);

			difference -= start_time;

			unsigned bin = difference.get_real_secs() / TOKENIZER_BIN_LENGTH;

//			cout << "difference: " << difference << " bin " << bin << " symbol: " << (int) peak.symbol << " sample_x: " << peak.sample_x << " channel " << (int) peak.channel << " fbin " << (int) peak.fbin << endl;

			if( bin > PIPELINE_BINS )
			{

//				cout << " trying to use too large of a bin, stopping..." << endl;
				continue;
			}

			// check to see if this timestamp is the newest we've ever seen
			// note this check uses the inaccurate get_real_secs() however this is ok for just a check
			if( peak_timestamp.get_real_secs() > newest_time.get_real_secs() )
			{
				newest_time = peak_timestamp;
				//				cout << "newer time" << endl;
			}


			// read bin
			boost::tuple<unsigned short, const PopPeak*> contents = symbols[peak.basestation][bin];

//			cout << "existing count " << contents.get<0>() << endl;

			// bump counter
			contents.get<0>()++;

			// only assign pointer into bin if not filled
			if( contents.get<1>() == NULL )
				contents.get<1>() = data + i;

			// save back to bin
			symbols[peak.basestation][bin] = contents;
//			cout << "got " << peak.basestation << " - " << (int) peak.symbol << peak.data[PEAK_SINC_NEIGHBORS+1].timestamp << endl;
		}


		PopTimestamp compare = PopTimestamp( newest_time );
		compare -= start_time;

		unsigned newest_sample_bin = bins_for_time( compare );


		// calculate the largest bin we are allowed to comb by
		// adding the end of the comb width to the pipeline delay
		unsigned largest_combable_bin = bins_for_bytes(sizeof(PHY_MSG_HELO));
		largest_combable_bin += bins_for_time( PIPELINE_DELAY );


		while( ( largest_combable_bin + last_comb ) < newest_sample_bin )
		{

			// debug force to base_station 1
			comb(last_comb, 0);
//			cout << "comb at " << count << endl;

			last_comb++;


		}

//		cout << "newest_sample_bin: " << newest_sample_bin << " largest_combable_bin: " << largest_combable_bin << endl;



//		if( compare.get_real_secs() > PIPELINE_DELAY )
//		{
//			// debug force to base_station 1
//			comb(0, 1);
//
//			cout << "comb at " << count << endl;
//		}

		// magic number launch
		if( count == 7156 )
		{
//			tokenize();
//
//			debug_print(0);
//			comb(26975, 1);
//while(1){};

			exit(0);

		}


	}

	// how many bins does it take to represent N bits
	unsigned bins_for_bytes(unsigned bytes)
	{
		return EXPECTED_BIN_SEPARATION * bytes * 8;
	}

	// how many bins does it take to represent N seconds
	unsigned bins_for_time(double n)
	{
		return ceil(n * BIN_COUNT);
	}
	unsigned bins_for_time(PopTimestamp t)
	{
		return bins_for_time(t.get_real_secs());
	}



	void comb(unsigned start, unsigned bs)
	{

		// create reference for easy syntax below
		vector<boost::tuple<unsigned short, const PopPeak*> > &s = symbols[bs];

		unsigned tally = 0;

//		cout << "comb from " << start << " to " << start + sizeof(PHY_MSG_HELO) * 8 * EXPECTED_BIN_SEPARATION << endl;

		for(unsigned i = 0; i < sizeof(PHY_MSG_HELO) * 8 * EXPECTED_BIN_SEPARATION ; i += EXPECTED_BIN_SEPARATION )
		{
			if( s[ start + i ].get<0>() != 0 || s[ start + i + 1].get<0>() != 0 )
			{
				tally++;
			}
		}




		if( tally >= 10 )
		{
			cout << "the comb says it takes " << tally << " bits to get to the center of the pop message" << endl;
			print_message( start, bs );
			cout << endl << endl;
		}

	}

	void print_message(unsigned start, unsigned bs)
	{
		// create reference for easy syntax below
		vector<boost::tuple<unsigned short, const PopPeak*> > &s = symbols[bs];

		unsigned tally = 0;

//		debug_print(1);

		cout << "Message at (" << start << ") ------- " << endl;

		for(unsigned i = 0; i < sizeof(PHY_MSG_HELO) * 8 * EXPECTED_BIN_SEPARATION ; i += EXPECTED_BIN_SEPARATION )
		{
//			boost::tuple<unsigned short, const PopPeak*> cur = symbols[bs][i+start];

			if( s[ start + i ].get<0>() != 0 )
			{
				cout << "  " << (int) s[ start + i ].get<1>()->symbol << endl;
				continue;
			}

//			boost::tuple<unsigned short, const PopPeak*> nxt = symbols[bs][i+start];

			if( s[ start + i + 1].get<0>() != 0 )
			{
				cout << "  " << (int) s[ start + i + 1].get<1>()->symbol << endl;
				continue;
			}

			cout << "   x" << endl;
		}
	}

	void debug_print(unsigned bs)
	{
		for(unsigned i = 0; i < PIPELINE_BINS; i++)
		{
			boost::tuple<unsigned short, const PopPeak*> contents = symbols[bs][i];

			if( contents.get<0>() != 0 )
				cout << "bin: " << i << " count: " << contents.get<0>() << " " << contents.get<1>() << endl;
		}
	}

	void tokenize()
	{
//
//		// sort Symbols in order
//		std::sort(symbols.begin(), symbols.end(), PopSymbol::timestamp_comparitor);
//
//		for( unsigned i = 0; i < symbols.size() - 1; i++ )
//		{
//			PopSymbol *current = &symbols[i];
//			PopSymbol *next = &symbols[i+1];
//
//			if( abs(current->timestamp.get_real_secs() - next->timestamp.get_real_secs()) < 0.001 )
//			{
//				PopTimestamp difference;// = PopTimestamp(current);
//
//				if( current->timestamp.get_real_secs() < next->timestamp.get_real_secs() )
//				{
//					difference = PopTimestamp(next->timestamp);
//					difference -= current->timestamp;
//
//					cout << "these two stamps (" << i << ", " << i+1 << ") have a difference of " << difference.get_frac_secs() << endl;
//					current->debug_print();
//					next->debug_print();
//
//				}
//
//
//
//			}
//			else
//			{
////				cout << "these two stamps (" << i << ", " << i+1 << ") are too far apart: " << endl;
////				current->debug_print();
////				next->debug_print();
////
////				cout << endl << endl;
//			}
//
////			PopTimestamp difference = PopTimestamp
//
//		}
//
//		// delete all but the last symbol
//		int erase = symbols.size() - 1;
//
//		// delete vector elements
//		for( int i = 0; i < erase; i++ )
//			symbols.erase(symbols.begin());
	}




};
} // namespace pop


#endif
