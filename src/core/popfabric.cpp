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

//FIXME: add simple routing to not send messages back to sender during a send_down()

PopFabric::PopFabric(zmq::context_t& context, std::string name, bool r, std::string ip_up) : fp(0), router(r), router_has_up(0), pub_up(0), sub_up(0), pub_down(0), sub_down(0)
{
	names.push_back(name);

	// bind to ports
	if( router )
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

		std::string bind_address("*");


		// listen for pubs on pub port
		pub_down = new zmq::socket_t(context, ZMQ_PUB);
		pub_down->bind(std::string("tcp://" + bind_address + ":" + FABRIC_PORT_PUB).c_str());

		// listen for subs on sub port
		sub_down = new zmq::socket_t(context, ZMQ_SUB);
		sub_down->bind(std::string("tcp://" + bind_address + ":" + FABRIC_PORT_SUB).c_str());
		sub_down->setsockopt( ZMQ_SUBSCRIBE, "_", 1); // subscribe to every message
	}

	// connect to up
	if( !router || router_has_up )
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

unsigned PopFabric::poll_downwards()
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

			if( std::find(names.begin(), names.end(), to) != names.end() )
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
			std::cout << "(" << names[0] << ") Received [" << to << "," << from << "] " << contents << std::endl;
#endif

			updates++;
		}
	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

unsigned PopFabric::poll_upwards()
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

			if( std::find(names.begin(), names.end(), to) != names.end() )
			{
				// message is for us
				if( this->fp )
				{
					this->fp(to, from, contents);
				}
			}
			else
			{
				if( router )
				{
					// route
					send_down(to, from, contents);

					// we do not send up because the message came from up
				}
			}

#ifdef FABRIC_VERBOSE
			std::cout << "(" << names[0] << ") Received [" << to << "," << from << "] " << contents << std::endl;
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
	cout << "(" << names[0] << ") Send Up: [" << to << "," << from << "] " << message << std::endl;
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
	cout << "(" << names[0] << ") Send Down: [" << to << "," << from << "] " << message << std::endl;
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
			send_up(to, names[0], message);
		}
		send_down(to, names[0], message);
	}
	else
	{
		send_up(to, names[0], message); // send from us
	}
}

void PopFabric::keepalive()
{
	std::string msg = "{}";
	std::string to = "KEEPALIVE";

	send(to, msg);
}

// returns number of (valid or invalid) updates / messages received during poll
unsigned PopFabric::poll()
{
	if( router )
	{
		unsigned count = 0;

		if( router_has_up )
		{
			count += poll_upwards();
		}
		count += poll_downwards();

		return count;
	}
	else
	{
		return poll_upwards();
	}
}

// Poll in a loop until poll reports that 0 messages were processed.
unsigned PopFabric::poll_burst(unsigned max)
{
	unsigned ret = 0;
	unsigned count = 0;
	for(unsigned i = 0; i < max; i++)
	{
		count = poll();

		// keep running tally
		ret += count;

		if(!count)
		{
			return ret;
		}
	}

	return ret;
}

void PopFabric::set_receive_function(boost::function<void(std::string, std::string, std::string)> in)
{
	this->fp = in;
}

void PopFabric::add_name(std::string name)
{
	if( std::find(names.begin(), names.end(), name) == names.end() )
	{
		names.push_back(name);

		ostringstream os;
		os << "{\"method\":\"node_broadcast\",\"params\":[\"" << name << "\", \"" << names[0] << "\"]}";
		send("noc", os.str());
	}
}


}
