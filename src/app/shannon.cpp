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

#include "net/popnetwork.hpp"
#include "sdr/popuhd.hpp"
#include "examples/popexamples.hpp"
#include "core/popsignal.hpp"
#include "dsp/prota/popprotadespread.hpp"
#include "dsp/prota/popprotatdmabin.hpp"

using namespace boost;
using namespace pop;
using namespace std;

namespace po = boost::program_options;

int test()
{
	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;
	unsigned server_port;
	string server_name;

	cout << "Shannon - PopWi Digital Signal Processing (DSP) Core" << endl;
	cout << "Copyright (c) 2013. PopWi Technology Group, Inc." << endl << endl;

	po::options_description desc("Shannon Command-line Options");
	desc.add_options()
	    ("help", "help message")
	    ("server", po::value<string>(&server_name)->default_value("papa.popwi.com"), "Remote Manager Location")
	    ("file", po::value<string>(&server_name)->default_value("shannon.xml"), "Setup File")
	    ("server-port", po::value<unsigned>(&server_port)->default_value(35005), "Incoming UDP port")
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

#if 0
	// test code
	PopPiSource pisource;
	PopTest1 test;

	pisource.connect(test);

	pisource.start();
#endif

#if 1
	PopProtATdmaBin bin;
	PopSlot slot;

	bin.connect(slot);

	bin.start();
#endif

#if 0
	// Initialize Graphics Card
	PopGpu popgpu;
	popgpu.start_thread();

	// Initialize Software Defined Radio (SDR) and start
	PopUhd popuhd;

	// Initialize Network Connection
	PopNetwork popnetwork;

	// Initialize Magnitude Block
	PopMagnitude popmag;

	popuhd.connect(popgpu);

	//popuhd.connect(popmag);

	popgpu.connect(popnetwork);

	//popmag.connect(popnetwork);

#endif


	// Run Control Loop
	while(1)
	{
		boost::posix_time::seconds workTime(1);
		boost::this_thread::sleep(workTime);
	}

    return ret;
}
