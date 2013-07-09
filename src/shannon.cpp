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

#include "popcontrol.hpp"
#include "popnetwork.hpp"
#include "popsdr.hpp"
#include "popgpu.hpp"
#include "popbin.hpp"
//#include "popdecimate.hpp"
//#include "popdata.hpp"
#include "popobject.hpp"
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


	// Initialize Graphics Card
	PopGpu popgpu;

	// Initialize Software Defined Radio (SDR) and start
	PopSdr popsdr;

	// Initialize Network Connection
	PopNetwork popnetwork;

	// Initialize Dummy Load
	PopDummySink dummysink;




	popsdr.connect(dummysink);
	popsdr.connect(popgpu);
	popgpu.connect(popnetwork);



	// Run Control Loop
	while(1) {}

    return ret;
}
