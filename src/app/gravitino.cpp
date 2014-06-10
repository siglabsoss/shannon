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
//#include "zhelpers.hpp"

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
	using namespace rbx;




//	 //  Prepare our context and subscriber
//	    zmq::context_t context(1);
//	    zmq::socket_t subscriber (context, ZMQ_SUB);
//	    subscriber.connect("tcp://localhost:5563");
//	    subscriber.setsockopt( ZMQ_SUBSCRIBE, "B", 1);
//
//	    while (1) {
//
//	        //  Read envelope with address
//	        std::string address = s_recv (subscriber);
//	        //  Read message contents
//	        std::string contents = s_recv (subscriber);
//
//	        std::cout << "[" << address << "] " << contents << std::endl;
//	    }
//	    return 0;
//

	cout << "grav awake" << endl;
	zmq::context_t context(1); // only 1 per thread
	PopChannelMap channel_map(false, context);





	Config::loadFromDisk();






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



	char c;

	int i = 0;

	// Run Control Loop
	while(1)
	{
		/*c = getch();
		if( c == '-' ) h_start_chan--;
		if( c == '+' ) h_start_chan++;*/

		// if( (c == '-') || (c == '+')) printf("h_start_chan = %lu\r\n", h_start_chan);
		boost::posix_time::milliseconds workTime(100);
		boost::this_thread::sleep(workTime);

		channel_map.poll();

		cout << "poll" << endl;

//		if( i % 1000 == 0)
//			file.read(1);

		i++;
	}

    return 0;
}
