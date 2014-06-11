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
#include "net/popnetwork.hpp"
//#include "mdl/poppeak.hpp"
//#include "core/simulateartemis.hpp"
#include "core/popserial.hpp"
#include "core/popgpsdevice.hpp"
#include "core/popartemisrpc.hpp"
#include "core/pops3prpc.hpp"
#include "core/popparsegps.hpp"
#include "core/poppackethandler.hpp"
#include "core/popchannelmap.hpp"
#include "core/utilities.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;



	Config::loadFromDisk();






	zmq::context_t context(1); // only 1 per thread

	PopChannelMap channel_map(false, context);


	PopArtemisRPC rpc(1);

	PopSerial uart0("/dev/ttyUSB0", 1000000);

	uart0.rx.connect(rpc);
	rpc.tx.connect(uart0);
	uart0.rx.start_thread();

	// Send a "set_role_base_station" RPC to the Artemis board to force it into
	// base station mode.
	rpc.set_role_base_station();

	PopPacketHandler handler(1);
	rpc.handler = &handler;
	handler.rpc = &rpc;


	PopParseGPS gps(1);
//	PopSerial uart4("/dev/tty4", 4800);
//	uart4.rx.connect(gps);
//	uart4.rx.start_thread();

	PopNetwork<char> json(0, Config::get<std::string>("basestation_s3p_ip"), Config::get<int>("basestation_s3p_port"), 0);

	PopS3pRPC s3p(0);
	handler.s3p = &s3p;

	s3p.tx.connect(json);
	json.connect(s3p);

//	PopGpsDevice updates(1);

//	rpc.packets.connect(updates);
//	updates.tx.connect(json);
//	updates.gps = &gps;
	json.wakeup();
//	updates.tx.start_thread();
//	handler.s3p = &updates;

	s3p.greet_s3p();

//	rpc.mock();




//	SimulateArtemis simArt(0);
//	simArt.rx.connect(rpc);
//	rpc.rx.connect(simArt);

//	simArt.rx.start_thread();
	//channel_map.set(i%POP_SLOT_COUNT, 54, 0);

	sleep(1);
	channel_map.poll();
	channel_map.checksum_dump();


	char c;

	int i = 0;

	// Run Control Loop
	while(1)
	{
		channel_map.poll();

		boost::posix_time::milliseconds workTime(1000);
		boost::this_thread::sleep(workTime);

		i++;
	}

    return 0;
}
