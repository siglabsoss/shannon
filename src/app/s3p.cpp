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
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sstream>


#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include "core/objectstash.hpp"
#include "core/poptokenizer.hpp"
#include "net/popnetworkjson.hpp"
//#include "sdr/popuhd.hpp"
#include "core/config.hpp"
#include "examples/popexamples.hpp"
#include "dsp/prota/popprotatdmabin.hpp"
#include "net/popnetwork.hpp"
#include "net/popwebhook.hpp"
#include "mdl/poppeak.hpp"
#include "core/geohelper.hpp"
#include "core/popbuchermultilateration.hpp"
#include "core/popgeolocation.hpp"
#include "core/popgravitinoparser.hpp"
#include "core/popsightingstore.hpp"
#include "core/poptrackerlocationstore.hpp"
#include "core/popchannelmap.hpp"
#include "core/popfabric.hpp"
#include "core/utilities.hpp"

//#include "core/popsourcemsg.hpp"

using namespace boost;
using namespace pop;
using namespace std;

namespace po = boost::program_options;

extern size_t h_start_chan;

int main(int argc, char *argv[])
{
	int ret = 0;
	unsigned incoming_port, outgoing_port;
	string incoming_address, outgoing_address;
	string server_name;
	string debug_file;

	cout << "Shannon - PopWi Server Side Signal Processing (S3P) Core" << endl;
	cout << "Copyright (c) 2013. PopWi Technology Group, Inc." << endl << endl;

	po::options_description desc("Shannon Command-line Options");
	desc.add_options()
	    ("help", "help message")
	    ("server", po::value<string>(&server_name)->default_value("papa.popwi.com"), "Remote Manager Location")
	    ("file", po::value<string>(&server_name)->default_value("shannon.xml"), "Setup File")
	    ("incoming-address", po::value<string>(&incoming_address)->default_value("173.167.119.220"), "Incoming UDP address")
	    ("incoming-port", po::value<unsigned>(&incoming_port)->default_value(5004), "Incoming UDP port")
	    ("outgoing-address", po::value<string>(&outgoing_address)->default_value("173.167.119.220"), "Outgoing UDP address")
	    ("outgoing-port", po::value<unsigned>(&outgoing_port)->default_value(35005), "Outgoing UDP port")
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

	zmq::context_t context(1); // only 1 per thread

	PopChannelMap channel_map("", true, context);





//	// Build Stash of PopRadio objects
//	ObjectStash stash;
//
//	// Populate with N Radio
//
//	int testRadioCount = 100000;
//
//	buildNFakePopRadios(stash, testRadioCount);
//
//	// This source prints to stdout
//	PopPrintCharStream printer;
//
//	// This source generates GPS changes for devices
//	PopRandomMoveGPS randomMove;
//
//	// wire in pointer to our global PopRadio stash
//	randomMove.stash = &stash;
//
//	// tell it which serial numbers to nudge around the map
//	randomMove.testRadioCount = testRadioCount;
//
//	// connect a source which prints
//	randomMove.connect(printer);
//
//	// start it up
//	randomMove.start();

	Config::loadFromDisk();

//	PopReadFromFile<PopPeak> file ("incoming_packets.raw");

//	PopDumpToFile<PopPeak> dump ("incoming_packets.raw");


	PopTokenizer tokenizer;

	PopWebhook hook(0);
	hook.init();

	GeoHelper geo_helper;
	PopBucherMultilateration multilateration(&geo_helper);
	PopGeoLocation geo_location(&geo_helper, &multilateration);
	PopTrackerLocationStore tracker_location_store(&hook);
	PopSightingStore sighting_store(&geo_location, &tracker_location_store);

	// there is only one s3p.  name discovery is a problem we will solve later
	PopFabric fabric(context, "s3p", true);

	PopGravitinoParser gravitinoParser(&sighting_store, &fabric);

//	file.connect(tokenizer);

	char c;

	int i = 0;

	// Run Control Loop
	while(1)
	{
		channel_map.poll();
		fabric.poll();

		boost::posix_time::milliseconds workTime(100);
		boost::this_thread::sleep(workTime);

		if( i % 30 == 0 )
		{
			channel_map.checksum_dump();
		}

		i++;
	}

    return ret;
}
