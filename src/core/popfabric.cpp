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


//#define FABRIC_VERBOSE

#define FABRIC_PORT_PUB   "11528"
#define FABRIC_PORT_SUB   "11529"

namespace pop
{

//FIXME: up connection in router mode isn't tested

PopFabric::PopFabric(zmq::context_t& context, std::string n, bool r, std::string ip_up) : fp(0), router(r), router_has_up(0), pub_up(0), sub_up(0), pub_down(0), sub_down(0), name(n)
{

	// bind to ports
	if( r )
	{
		// this bool is only set if in router mode, and we've connected an up
		if(ip_up.compare("null") == 0 || ip_up.compare("") == 0)
		{
			router_has_up = 0;
		}
		else
		{
			router_has_up = 1;
		}


		// listen for pubs on pub port
		pub_down = new zmq::socket_t(context, ZMQ_PUB);
		pub_down->bind("tcp://*:" FABRIC_PORT_PUB);

		// listen for subs on sub port
		sub_down = new zmq::socket_t(context, ZMQ_SUB);
		sub_down->bind("tcp://*:" FABRIC_PORT_SUB);
		sub_down->setsockopt( ZMQ_SUBSCRIBE, "_", 1); // subscribe to every message
	}

	// connect to up
	if( !r || router_has_up )
	{
		// my pub connects to your sub (port)
		pub_up = new zmq::socket_t(context, ZMQ_PUB);
		pub_up->connect(std::string("tcp://" + ip_up + ":" + FABRIC_PORT_SUB).c_str());

		// my sub connects to your pub (port)
		sub_up = new zmq::socket_t(context, ZMQ_SUB);
		sub_up->connect(std::string("tcp://" + ip_up + ":" + FABRIC_PORT_PUB).c_str());
		sub_up->setsockopt( ZMQ_SUBSCRIBE, "_", 1); // subscribe to every message
	}
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
					this->fp(to, from, contents);
				}
			}
			else
			{
				// route
				send_down(to, from, contents);

				if( router_has_up )
				{
					send_up(to, from, contents);
				}
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






	unsigned updates = 0;

	zmq::pollitem_t items [] = {
			{ *sub_up, 0, ZMQ_POLLIN, 0 }
	};

	zmq::message_t message;

	do {
		// items, number of items in array, timeout (-1 is block forever)
		zmq::poll (items, 1, 0);

		if (items[0].revents & ZMQ_POLLIN) {

			//  Read message filter which is always _
			std::string underscore = s_recv(*sub_up);
			if( underscore.compare("_") != 0 )
			{
				// parsing for _ allows for messaging re-syncing
				continue;
			}


			// read all parts, never abort early as this will cause sync issues in the stream
			std::string to = s_recv(*sub_up);
			std::string from = s_recv(*sub_up);
			std::string contents = s_recv(*sub_up);

			if( to.compare(name) == 0 )
			{
				// message is for us
				if( this->fp )
				{
					this->fp(to, from, contents);
				}
			}
			else
			{
				// ignore b/c we do not route
			}

#ifdef FABRIC_VERBOSE
			std::cout << "[" << to << "," << from << "] " << contents << std::endl;
#endif

			updates++;
		}
	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

void PopFabric::send_up(std::string to, std::string from, std::string message)
{
	if( pub_up )
	{
#ifdef FABRIC_VERBOSE
	cout << "Send Up: [" << to << "," << from << "] " << message << std::endl;
#endif
		// sending an _ allows for messaging re-syncing (if subscriber doesn't pull each piece correctly)
		s_sendmore(*pub_up, std::string("_"));
		s_sendmore(*pub_up, to);
		s_sendmore(*pub_up, from);
		s_send(*pub_up,     message);
	}
	else
	{
		cout << "trying to send_up() when no upstream zmq connection exists" << endl;
	}
}

void PopFabric::send_down(std::string to, std::string from, std::string message)
{
#ifdef FABRIC_VERBOSE
	cout << "Send Down: [" << to << "," << from << "] " << message << std::endl;
#endif
	s_sendmore(*pub_down, std::string("_"));
	s_sendmore(*pub_down, to);
	s_sendmore(*pub_down, from);
	s_send(*pub_down,     message);
}

void PopFabric::send(std::string to, std::string message)
{
	if(router)
	{
		if( router_has_up )
		{
			send_up(to, name, message);
		}
		send_down(to, name, message);
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

void PopFabric::set_receive_function(boost::function<void(std::string, std::string, std::string)> in)
{
	this->fp = in;
}


}