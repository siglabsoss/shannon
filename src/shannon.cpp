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

#include <popnetwork.hpp>
#include <popsdr.hpp>
#include <popgpu.hpp>
#include <popexamples.hpp>

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
	unsigned inport, outport;

	cout << "Shannon - Base station Digital Signal Processing (DSP) Core" << endl;
	cout << "Copyright (c) 2013. PopWi Technology Group, Inc." << endl << endl;

	po::options_description desc("Shannon Command-line Options");
	desc.add_options()
	    ("help", "help message")
	    ("inport", po::value<unsigned>(&inport)->default_value(5004), "Incoming UDP port")
	    ("outport", po::value<unsigned>(&outport)->default_value(35005), "Outgoing UDP port")
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
	// Initialize Graphics Card
	PopGpu popgpu;
	popgpu.start_thread();

	// Initialize Software Defined Radio (SDR) and start
	PopSdr popsdr;

	// Initialize Network Connection
	PopNetwork popnetwork;

	// Initialize Magnitude Block
	PopMagnitude popmag;

	popsdr.connect(popgpu);

	//popsdr.connect(popmag);

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
