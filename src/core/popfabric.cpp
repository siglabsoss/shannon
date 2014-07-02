/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <stddef.h>

#include <vector>

#include <boost/thread/mutex.hpp>

#include "core/popfabric.hpp"
#include "zmq/zhelpers.hpp"
#include "frozen/frozen.h"
#include "core/utilities.hpp"
#include "core/poppackethandler.hpp"

using boost::mutex;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;
using namespace zmq;
using namespace std;

//#define POP_CHANNEL_MAP_TOKENS (12)

#define FABRIC_VERBOSE

#define FABRIC_PORT_PUB_UP   "11528"
#define FABRIC_PORT_SUB_UP   "11529"
#define FABRIC_PORT_PUB_DOWN "11530"
#define FABRIC_PORT_SUB_DOWN "11531"

namespace pop
{

// FIXME: copy the logic of http://zguide.zeromq.org/php:chapter5#Reliable-Pub-Sub-Clone-Pattern


PopFabric::PopFabric(zmq::context_t& context, std::string n, bool r, std::string ip_up) : fp(0), router(r), pub_up(0), sub_up(0), pub_down(0), sub_down(0), name(n)
{
	if( r )
	{
//		pub_down = new zmq::socket_t(context, ZMQ_PUB);
//		pub_down->bind("tcp://*:" FABRIC_PORT_SUB_UP);

		sub_down = new zmq::socket_t(context, ZMQ_SUB);
		sub_down->bind("tcp://*:" FABRIC_PORT_PUB_UP);
		sub_down->setsockopt( ZMQ_SUBSCRIBE, "_", 1); // subscribe to every message

//		sub_up = new zmq::socket_t(context, ZMQ_SUB);
//		sub_up->connect(std::string("tcp://" + ip + ":" + FABRIC_PORT_SUB_UP).c_str());
	}
	else
	{
		pub_up = new zmq::socket_t(context, ZMQ_PUB);
		pub_up->connect(std::string("tcp://" + ip_up + ":" + FABRIC_PORT_PUB_UP).c_str());

//		sub_up = new zmq::socket_t(context, ZMQ_SUB);
//		sub_up->connect(std::string("tcp://" + ip_up + ":" + FABRIC_PORT_SUB_UP).c_str());
	}

	cout << name << endl;

	//	if( master )
	//	{
	//		publisher = new zmq::socket_t(context, ZMQ_PUB);
	//		publisher->bind("tcp://*:11526");
	//		collector = new zmq::socket_t(context, ZMQ_PULL);
	//		collector->bind("tcp://*:11527");
	//	}
	//	else
	//	{
	//		subscriber = new zmq::socket_t(context, ZMQ_SUB);
	//		subscriber->connect(std::string("tcp://" + ip + ":11526").c_str());
	//		subscriber->setsockopt( ZMQ_SUBSCRIBE, "CHANNEL_MAP", 11);
	//		pusher = new zmq::socket_t(context, ZMQ_PUSH);
	//		pusher->connect(std::string("tcp://" + ip + ":11527").c_str());
	//
	//		request_sync();
	//	}
}


PopFabric::~PopFabric()
{
//	mutex::scoped_lock lock(mtx_);
//
//	if( publisher )
//	{
//		delete publisher;
//	}
//
//	if( collector )
//	{
//		delete collector;
//	}
//
//	if( subscriber )
//	{
//		delete subscriber;
//	}
//
//	if( pusher )
//	{
//		delete pusher;
//	}

	// Clean up the memory used by the map values.
//	for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end();
//		 ++it) {
//		delete it->second;
//	}

}

unsigned PopFabric::router_poll()
{
	unsigned updates = 0;

	zmq::pollitem_t items [] = {
			{ *sub_down, 0, ZMQ_POLLIN, 0 }
	};

	zmq::message_t message;

	do {
		// items, number of items in array, timeout (-1 is block forever)
		zmq::poll (items, 1, 0);

		if (items[0].revents & ZMQ_POLLIN) {

			//  Read message filter which is always _
			std::string underscore = s_recv(*sub_down);
			if( underscore.compare("_") != 0 )
			{
				// parsing for _ allows for messaging re-syncing
				continue;
			}


			// read all parts, never abort early as this will cause sync issues in the stream
			std::string to = s_recv(*sub_down);
			std::string from = s_recv(*sub_down);
			std::string contents = s_recv(*sub_down);

			if( to.compare(name) == 0 )
			{
				// message is for us
				if( this->fp )
				{
					this->fp(from, contents);
				}
			}
			else
			{
				// route
			}

#ifdef FABRIC_VERBOSE
			std::cout << "[" << to << "," << from << "] " << contents << std::endl;
#endif

			updates++;
		}
	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

unsigned PopFabric::node_poll()
{
	unsigned updates = 0;






	//	zmq::pollitem_t items [] = {
	//				{ *collector, 0, ZMQ_POLLIN, 0 },
	//				{ *subscriber, 0, ZMQ_POLLIN, 0 }
	//		};
	//
	//		zmq::message_t message;
	//
	//		do {
	//
	//			// items, number of items in array, timeout (-1 is block forever)
	//			zmq::poll (items, 2, 0);
	//
	//			if (items[1].revents & ZMQ_POLLIN) {
	//				//  Read envelope with address
	//				std::string address = s_recv(*subscriber);
	//				//  Read message contents
	//				std::string contents = s_recv(*subscriber);
	//
	//				std::cout << "[" << address << "] " << contents << std::endl;
	//				//  Process weather update
	//			}
	//		} while(items[0].revents & ZMQ_POLLIN);










//	zmq::pollitem_t items [] = {
//			{ *subscriber, 0, ZMQ_POLLIN, 0 }
//	};
//
//	zmq::message_t message;
//
//	do {
//		// items, number of items in array, timeout (-1 is block forever)
//		zmq::poll (items, 1, 0);
//
//		if (items[0].revents & ZMQ_POLLIN) {
//			//  Read message filter
//			std::string filter = s_recv(*subscriber);
//
//			//  Read message contents
//			std::string contents = s_recv(*subscriber);
//
//#ifdef FABRIC_VERBOSE
//			std::cout << "[" << filter << "] " << contents << std::endl;
//#endif
//
//			patch_datastore(contents);
//
//			updates++;
//		}
//	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

void PopFabric::send_up(std::string to, std::string from, std::string message)
{
	// sending an _ allows for messaging re-syncing (if subscriber doesn't pull each piece correctly)
	s_sendmore(*pub_up, std::string("_"));
	s_sendmore(*pub_up, to);
	s_sendmore(*pub_up, from);
	s_send(*pub_up,     message);
}

void PopFabric::send_down(std::string to, std::string from, std::string message)
{
//	s_sendmore(*pub_up, std::string("_"));
//	s_sendmore(*pub_up, to);
//	s_sendmore(*pub_up, from);
//	s_send(*pub_up,     message);
}

void PopFabric::send(std::string to, std::string message)
{
	if(router)
	{

	}
	else
	{
		send_up(to, name, message); // send from us
	}
}

// returns number of (valid or invalid) updates / messages received during poll
unsigned PopFabric::poll()
{
	if( router )
	{
		return router_poll();
	}
	else
	{
		return node_poll();
	}
}

void PopFabric::set_receive_function(boost::function<void(std::string, std::string)> in)
{
	this->fp = in;
}


}
