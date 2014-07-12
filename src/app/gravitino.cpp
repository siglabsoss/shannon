/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>
//#include <complex>

//#include <boost/bind.hpp>
//#include <boost/program_options.hpp>

//#include "core/objectstash.hpp"
//#include "core/poptokenizer.hpp"
//#include "net/popnetworkjson.hpp"
#include "core/config.hpp"
//#include "examples/popexamples.hpp"
//#include "dsp/prota/popprotatdmabin.hpp"
//#include "mdl/poppeak.hpp"
//#include "core/simulateartemis.hpp"
#include "core/popserial.hpp"
#include "core/popgpsdevice.hpp"
#include "core/popartemisrpc.hpp"
#include "core/pops3prpc.hpp"
#include "core/popparsegps.hpp"
#include "core/poppackethandler.hpp"
#include "core/popchannelmap.hpp"
#include "core/popfabric.hpp"
#include "core/popfabricbridge.hpp"
#include "core/utilities.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;



	Config::loadFromDisk();






	zmq::context_t context(1); // only 1 per thread
	PopChannelMap channel_map(Config::get<std::string>("basestation_s3p_ip"), false, context);

	cout << "Waiting for Channel Map to sync" << endl;

	while(channel_map.dirty())
	{
		boost::posix_time::milliseconds workTime(500);
		boost::this_thread::sleep(workTime);
		cout << '.';
		channel_map.poll();
	}

	cout << "  done!" << endl;

	PopFabric basestation_fabric(context, pop_get_hostname(), true, Config::get<std::string>("basestation_s3p_ip"));

	uint32_t target_slots = (POP_SLOT_COUNT/4);
	uint32_t owned_slots = channel_map.allocated_count();
	if( target_slots > owned_slots )
	{
		// only request slots to meet our target
		channel_map.request_block(target_slots-owned_slots);
	}

	std::string attached_uuid;

	// reset device at baud 1000000
	{
		PopArtemisRPC rpc(NULL);
		PopSerial uart0("/dev/ttyUSB0", 1000000, "one");
		rpc.tx.connect(uart0);
		rpc.send_reset();
	}

	// reset device at baud 115200
	{
		int j = 0;
		PopArtemisRPC rpc(NULL);
		PopSerial uart0("/dev/ttyUSB0", 115200, "two");
		rpc.tx.connect(uart0);
		uart0.rx.connect(rpc);
		rpc.send_reset();
		boost::posix_time::milliseconds one_second(1000);
		for(j = 0; j < 4; j++)
		{
			boost::this_thread::sleep(one_second);
			rpc.set_role_base_station();
		}

		boost::posix_time::milliseconds two_second(2000);
		boost::this_thread::sleep(two_second);

		if(!rpc.received_basestation_boot())
		{
			cout << endl << endl << "attached device did not startup in basestation mode!!!" << endl << endl;
			boost::this_thread::sleep(one_second);
		}
		else
		{
			attached_uuid = rpc.attached_uuid;
			cout << endl << endl << "Attached device started correctly with serial: " << attached_uuid << endl;

			ostringstream os;
			os << "{\"method\":\"log\",\"params\":[\"" << "Basestation: " << pop_get_hostname() << " started with attacehd device: " << attached_uuid << "\"]}";
			basestation_fabric.send("s3p", os.str());

			ostringstream os2;
			os2 << "{\"method\":\"node_broadcast\",\"params\":[\"" << attached_uuid << "\", \"" << pop_get_hostname() << "\"]}";
			basestation_fabric.send("noc", os2.str());

		}
	}


	PopFabric attached_device_fabric(context, attached_uuid, false, "localhost");
	PopSerial uart0("/dev/ttyUSB0", 1000000, "three");
	PopArtemisRPC rpc(&attached_device_fabric);
	uart0.rx.connect(rpc);
	rpc.tx.connect(uart0);


	// Send a "set_role_base_station" RPC to the Artemis board to force it into
	// base station mode.
	rpc.set_role_base_station();

	PopPacketHandler handler(1);
	rpc.handler = &handler;
	handler.rpc = &rpc;
	handler.map = &channel_map;


	PopParseGPS gps(1);
	PopSerial uart1("/dev/ttyUSB1", 4800, "gps");
	uart1.rx.connect(gps);
	gps.tx.connect(uart1);

	gps.set_debug_on();
	gps.hot_start();

	PopFabricBridge bridge(&basestation_fabric, "s3p");
	PopS3pRPC s3p(0);
	bridge.tx.connect(s3p);
	s3p.tx.connect(bridge);

//	handler.s3p = &s3p;
	s3p.greet_s3p();

	// greet NOC
	ostringstream os3;
	os3 << "{\"method\":\"node_broadcast\",\"params\":[\"" << pop_get_hostname() << "\", \"" << "s3p" << "\"]}";
	basestation_fabric.send("noc", os3.str());



	char c;

	int i = 0;

	int j = 0;

	// Run Control Loop
	while(1)
	{
		for(j=0; j < 3; j++)
		{
			channel_map.poll();
			basestation_fabric.poll();
			attached_device_fabric.poll();
		}


		boost::posix_time::milliseconds workTime(100);
		boost::this_thread::sleep(workTime);

		i++;
	}

    return 0;
}
