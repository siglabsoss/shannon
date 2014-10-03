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
#include "core/popled.hpp"
#include "core/utilities.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;



	Config::loadFromDisk();
	PopLED led;





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
	std::string artemis_uart_path = Config::get<std::string>("artemis_uart");

	bool restart_artemis = true;

	// check if device is already in booted and in basestation mode
	{
		PopArtemisRPC rpc(NULL);
		PopSerial uart0(artemis_uart_path, 1000000, "pinger");
		rpc.tx.connect(uart0);
		rpc.send_enable_dma_spool(0);
		uart0.rx.connect(rpc);
		rpc.send_ping();
		boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
		if( rpc.last_pong.get_real_secs() != 0.0 )
		{
			restart_artemis = false;
			attached_uuid = rpc.attached_uuid;
		}
	}

	// read from config file, with a default value
	{
	bool force_restart = Config::get<bool>("force_attached_artemis_restart", true);
	restart_artemis |= force_restart;
	}


	// reset device at baud 1000000
	if( restart_artemis )
	{
		PopArtemisRPC rpc(NULL);
		PopSerial uart0(artemis_uart_path, 1000000, "one");
		rpc.tx.connect(uart0);
		rpc.send_reset();
		boost::this_thread::sleep(boost::posix_time::milliseconds(10));
		rpc.send_reset();
	}

	// reset device at baud 115200  (double if() is intentional to create scoped PopSerial objects)
	if( restart_artemis )
	{
		int j = 0;
		PopArtemisRPC rpc(NULL);
		PopSerial uart0(artemis_uart_path, 115200, "two");
		rpc.tx.connect(uart0);
		uart0.rx.connect(rpc);
		rpc.send_reset();
		boost::posix_time::milliseconds one_second(1000);
		for(j = 0; j < 5; j++)
		{
			boost::this_thread::sleep(one_second);
			rpc.set_role_base_station();

			if( rpc.received_basestation_boot() )
			{
				break;
			}
		}

		if(!rpc.received_basestation_boot())
		{
			cout << endl << endl << "attached device did not startup in basestation mode!!!" << endl << endl;
			boost::this_thread::sleep(one_second);
		}
		else
		{
			attached_uuid = rpc.attached_uuid;
			cout << endl << endl << "Attached device started correctly with serial: " << attached_uuid << endl;

			ostringstream os2;
			os2 << "{\"method\":\"node_broadcast\",\"params\":[\"" << attached_uuid << "\", \"" << pop_get_hostname() << "\"]}";
			basestation_fabric.send("noc", os2.str());

		}
	}
	else
	{
		cout << endl << endl << "Attached device left running with serial: " << attached_uuid << endl;

		ostringstream os2;
		os2 << "{\"method\":\"node_broadcast\",\"params\":[\"" << attached_uuid << "\", \"" << pop_get_hostname() << "\"]}";
		basestation_fabric.send("noc", os2.str());
	}


	PopFabric attached_device_fabric(context, attached_uuid, false, "localhost");
	PopSerial uart0(artemis_uart_path, 1000000, "three", false);
	PopArtemisRPC rpc(&attached_device_fabric, attached_uuid);
	rpc.send_ping();
	rpc.led = &led;

#ifdef READ_MODE
	PopReadFromFile<char> file ("incoming_chars.raw");
	file.verbose = false;
	file.connect(rpc);
#else
	uart0.rx.connect(rpc);
	rpc.tx.connect(uart0);
#endif

	rpc.send_enable_dma_spool(1);



	// Send a "set_role_base_station" RPC to the Artemis board to force it into
	// base station mode.
	rpc.set_role_base_station();

//	ostringstream lna_setting;
//	lna_setting << "{\"method\":\"set_external_lna\",\"params\":[" << 3 << "]}";
//	rpc.send_rpc(lna_setting.str().c_str(), lna_setting.str().length());
//
//	ostringstream rx_thresh;
//	rx_thresh << "{\"method\":\"set_rx_threshold\",\"params\":[" << 255 << "]}";
//	rpc.send_rpc(rx_thresh.str().c_str(), rx_thresh.str().length());

	PopPacketHandler handler(1);
	rpc.handler = &handler;
	handler.rpc = &rpc;
	handler.map = &channel_map;
	handler.start_thread();
	rpc.edges.connect(handler);

//	rpc.mock();

	PopParseGPS gps(1);
	PopSerial *uart1;

	if( Config::get<bool>("enable_gps") )
	{
		std::string gps_path = Config::get<std::string>("gps_uart");


		uart1 = new PopSerial(gps_path, 4800, "gps");
		uart1->rx.connect(gps);
		gps.tx.connect(*uart1);

		gps.set_debug_on();
		gps.hot_start();
	}

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

		led.poll();
		rpc.poll();

		boost::posix_time::milliseconds workTime(100);
		boost::this_thread::sleep(workTime);

		i++;


#ifdef READ_MODE
		file.read(1200*4 * 100);
#endif
	}

    return 0;
}
