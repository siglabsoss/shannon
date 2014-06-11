/******************************************************************************
R* Copyright 2013 PopWi Technology Group, Inc. (PTG)
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

//#include "zhelpers.hpp"

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
#include "net/popnetworkwrapped.hpp"
#include "net/popwebhook.hpp"
#include "mdl/poppeak.hpp"
#include "core/popgravitinoparser.hpp"
#include "core/popmultilateration.hpp"
#include "core/popsightingstore.hpp"
#include "core/poptrackerlocationstore.hpp"
#include "core/popchannelmap.hpp"
#include "core/utilities.hpp"

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

	PopChannelMap channel_map(true, context);

//	cout << "after sleep " << endl;
//	sleep(5);
//	cout << "after sleep " << endl;

//	while(1)
//	{
//		channel_map.get_block(5);
//		channel_map.get_block(2);
//		channel_map.get_block(16);
//		channel_map.clear_map();
//		sleep(3);
//		channel_map.poll();
//	}
//	channel_map.get_block(1);
//	channel_map.get_block(1);
//	channel_map.get_block(1);
//	channel_map.get_block(1);








//	 //  Prepare our context and publisher
//	    zmq::context_t context(1);
//	    zmq::socket_t publisher(context, ZMQ_PUB);
//	    publisher.bind("tcp://*:5563");
//
//	    while (1) {
//	        //  Write two messages, each with an envelope and content
//	        s_sendmore (publisher, "A");
//	        s_send (publisher, "We don't want to see this B");
//	        s_sendmore (publisher, "B");
//	        s_send (publisher, "We would like to see this");
//	        sleep (1);
//	    }
//	    return 0;






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

	PopNetworkWrapped<char> basestationConnection(Config::get<int>("basestation_s3p_port"), "", 0);

	PopTokenizer tokenizer;

	PopMultilateration multilateration;
	PopTrackerLocationStore tracker_location_store;
	PopSightingStore sighting_store(&multilateration, &tracker_location_store);

	PopGravitinoParser gravitinoParser(0, &sighting_store);

	basestationConnection.connect(gravitinoParser);
	gravitinoParser.tx.connect(basestationConnection);

	// call this after connecting all sources or sinks
	basestationConnection.wakeup();

	//PopWebhook hook(0);

	//gravitinoParser.tx.connect(hook);

//	file.connect(tokenizer);
	//channel_map.get_block(5);

	char c;

		int i = 0, dump = -1, updates = 0;

		// Run Control Loop
		while(1)
		{
			if(kbhit())
			{
				c = getch();
				if( c == '\n' )
				{
					channel_map.get_block(5);
					dump = i + 3;
				}
			}
			updates = channel_map.poll();

			if( i == dump || updates != 0 )
			{
				channel_map.checksum_dump();
				updates = 0;
			}
			//		cout << "poll" << endl;


			boost::posix_time::milliseconds workTime(100);
			boost::this_thread::sleep(workTime);


			i++;
		}

		return 0;
	}
