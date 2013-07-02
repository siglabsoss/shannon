/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>

#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include "popcontrol.hpp"
#include "popnetwork.hpp"
#include "popsdr.hpp"
#include "popgpu.hpp"
#include "popbin.hpp"
#include "popdecimate.hpp"

using namespace boost;
using namespace pop;
using namespace std;

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
	int ret;
	unsigned inport, outport;

	cout << "Shannon - Base station Digital Signal Processing (DSP) Core" << endl;
	cout << "Copyright (c) 2013. PopWi Technology Group, Inc." << endl << endl;

	po::options_description desc("Shannon Command-line Options");
	desc.add_options()
	    ("help", "help message")
	    ("inport", po::value<unsigned>(&inport)->default_value(5004), "Incoming UDP port")
	    ("outport", po::value<unsigned>(&outport)->default_value(5005), "Outgoing UDP port")
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

	// Initialize Controller
	PopControl popcontrol;

	// Initialize Graphics Card
	PopGpu popgpu;

	// Initialize Software Defined Radio (SDR) and start
	PopSdr popsdr;

	// Initialize Binning Class
	PopBin popbin;

	// Initialize Decimating Class
	PopDecimate popdecimate(1);

	// Initialize Network With 5004 Incoming Port and 5005 Outgoing Port
	PopNetwork popnetwork(inport, outport);

	// Attach SDR signal to GPU
	popsdr.sig.connect(bind(&PopGpu::import, &popgpu, _1, _2));

	// Attach GPU to Network
	//popgpu.sig.connect(bind(&PopNetwork::send, &popnetwork, _1, _2));

	// Attach GPU to binner
	//popgpu.sig.connect(bind(&PopBin::import, &popbin, _1, _2));

	// Attach GPU to Decimator
	//popgpu.sig.connect(bind(&PopDecimate::import, &popdecimate, _1, _2));

	// Attacj GPU to Network
	popgpu.sig.connect(bind(&PopNetwork::send, &popnetwork, _1, _2));

	// Attach decimator output to Network
	//popdecimate.sig.connect(bind(&PopNetwork::send, &popnetwork, _1, _2));

	// Run Control Loop
	ret = popcontrol.run();

    return ret;
}
