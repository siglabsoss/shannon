/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>
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
#include "core/popjsonrpc.hpp"
#include "core/popparsegps.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;
	using namespace rbx;



	Config::loadFromDisk();






	PopJsonRPC rpc(1);

	PopSerial uart2("/dev/ttyO2");

	uart2.rx.connect(rpc);
	rpc.rx.connect(uart2);
	uart2.rx.start_thread();


	PopParseGPS gps(1);
	PopSerial uart4("/dev/ttyO4", 4800);
	uart4.rx.connect(gps);
	uart4.rx.start_thread();

	PopNetwork<char> json(0, Config::get<std::string>("basestation_s3p_ip"), Config::get<int>("basestation_s3p_port"), 1);

	PopGpsDevice fakeUpdates(1);

	fakeUpdates.tx.connect(json);
	fakeUpdates.gps = &gps;
	json.wakeup();
	fakeUpdates.tx.start_thread();





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
		boost::posix_time::milliseconds workTime(1000);
		boost::this_thread::sleep(workTime);

//		if( i % 1000 == 0)
//			file.read(1);

		i++;
	}

    return 0;
}
