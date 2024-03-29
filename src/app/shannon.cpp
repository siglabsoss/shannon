/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>
#include <complex>

#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include "net/popnetworkcomplex.hpp"
#include "sdr/popuhd.hpp"
#include "examples/popexamples.hpp"
#include "dsp/prota/popchanfilter.hpp"
#include "dsp/prota/popprotatdmabin.hpp"
#include "dsp/prota/popdeconvolve.hpp"
#include "core/config.hpp"
#include "net/popnetwork.hpp"
#include "mdl/popsymbol.hpp"
#include "core/poptimestampinterpolate.hpp"
#include "dsp/prota/popbinner.hpp"
#include "core/utilities.hpp"

#include "dsp/common/poputils.hpp"

//#include "core/popsourcemsg.hpp"

using namespace boost;
using namespace pop;
using namespace std;
using namespace rbx;

namespace po = boost::program_options;

extern size_t h_start_chan;

int main(int argc, char *argv[])
{
	int ret = 0;
	unsigned incoming_port, outgoing_port;
	string incoming_address, outgoing_address;
	string server_name;
	string debug_file;

	cout << "Shannon - PopWi Digital Signal Processing (DSP) Core" << endl;
	cout << "Copyright (c) 2013. PopWi Technology Group, Inc." << endl << endl;

	po::options_description desc("Shannon Command-line Options");
	desc.add_options()
	    ("help", "help message")
	    ("server", po::value<string>(&server_name)->default_value("papa.popwi.com"), "Remote Manager Location")
	    ("file", po::value<string>(&server_name)->default_value("shannon.xml"), "Setup File")
	    ("incoming-address", po::value<string>(&incoming_address)->default_value("127.0.0.1"), "Incoming UDP address")
	    ("incoming-port", po::value<unsigned>(&incoming_port)->default_value(5004), "Incoming UDP port")
	    ("outgoing-address", po::value<string>(&outgoing_address)->default_value("127.0.0.1"), "Outgoing UDP address")
	    ("outgoing-port", po::value<unsigned>(&outgoing_port)->default_value(5005), "Outgoing UDP port")
	    ("debug-file", po::value<string>(&debug_file)->default_value("dat/dump.raw"), "filename used for raw data dump")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if( vm.count("help") )
	{
		cout << endl << desc << endl;
		cout <<
		"    Command-line options override program defaults." << endl << endl;
		return ~0;
	}



#if 1

	Config::loadFromDisk();

	// Initialize Graphics Card
	PopChanFilter chanfilter;
	chanfilter.start_thread();

	// Initialize Protocol A bin
	//PopProtATdmaBin bin;
	
	// Initialize Software Defined Radio (SDR) and start
	PopUhd popuhd;

	// Initialize data upconversion
	PopTypeConversion<complex<float>, complex<double> > conv;

	// Initialize Decimator
	//PopDecimate<complex<float> > decimate(64);

	// Setup timestamp interpolate block.  This number is hardcoded.. how can we grab it from popuhd?
	PopTimestampInterpolation<complex<double> > timestampInterpolation(507);
	timestampInterpolation.start_thread();

	popuhd.connect(timestampInterpolation);

	timestampInterpolation.connect(chanfilter);

	PopProtADeconvolve deconvolve;
	deconvolve.start_thread();

//	chanfilter.strided_gpu.debug_free_buffers = true;

	chanfilter.strided_gpu.connect(deconvolve);
	//chanfilter.connect(popnetwork);

	PopBinner binner;
	binner.start_thread();

	deconvolve.cts_mag_gpu.connect(binner);

	// Open Network Connection to our designated s3p
//	PopNetwork<PopPeak> s3pConnection(0, Config::get<std::string>("basestation_s3p_ip"), Config::get<int>("basestation_s3p_port"), 1);

//	deconvolve.peaks.connect(s3pConnection);

	// call this after connecting all sources or sinks
//	s3pConnection.wakeup();
//	s3pConnection.process();

	//PopDumpToFile<complex<double> > dump;



	popuhd.start();

#endif

	char c;
	size_t i = 0;

	// Run Control Loop
	while(1)
	{
		/*c = getch();
		if( c == '-' ) h_start_chan--;
		if( c == '+' ) h_start_chan++;*/

		// if( (c == '-') || (c == '+')) printf("h_start_chan = %lu\r\n", h_start_chan);
		boost::posix_time::microseconds workTime(10);
		boost::this_thread::sleep(workTime);

		// check to see if the SDR is frozen
		if( i > 150000 && popuhd.init_stage < 4 )
		{
			if( i % 1000 == 0)
				cout << RED "ERROR: PopUhd looks frozen.  The SDR may require a restart!" RESETCOLOR << endl;
		}

		i++;
	}

    return ret;
}
