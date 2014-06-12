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
#include <stdint.h>
#include <time.h>

#include <map>
#include <string>
//#include <tr1/unordered_map>
#include <utility>
#include <vector>
#include <zmq.hpp>
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <sstream>

#include <boost/thread/mutex.hpp>

#include "dsp/prota/popsparsecorrelate.h"
#include "core/popchannelmap.hpp"
#include "zmq/zhelpers.hpp"
#include "frozen/frozen.h"
#include "core/utilities.hpp"

using boost::mutex;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;
using namespace zmq;
using namespace std;

#define POP_CHANNEL_MAP_TOKENS (12)

#define CHANNEL_MAP_VERBOSE

namespace pop
{

// FIXME: copy the logic of http://zguide.zeromq.org/php:chapter5#Reliable-Pub-Sub-Clone-Pattern

PopChannelMap::PopChannelMap(std::string ip, bool m, zmq::context_t& context) : master(m), dirty_(0), publisher(0), subscriber(0), pusher(0)
{
	if( master )
	{
		publisher = new zmq::socket_t(context, ZMQ_PUB);
		publisher->bind("tcp://*:11526");
		collector = new zmq::socket_t(context, ZMQ_PULL);
		collector->bind("tcp://*:11527");
	}
	else
	{
		subscriber = new zmq::socket_t(context, ZMQ_SUB);
		subscriber->connect(std::string("tcp://" + ip + ":11526").c_str());
		subscriber->setsockopt( ZMQ_SUBSCRIBE, "CHANNEL_MAP", 11);
		pusher = new zmq::socket_t(context, ZMQ_PUSH);
		pusher->connect(std::string("tcp://" + ip + ":11527").c_str());

		request_sync();
	}
}

PopChannelMap::~PopChannelMap()
{
	mutex::scoped_lock lock(mtx_);

	if( publisher )
	{
		delete publisher;
	}

	if( collector )
	{
		delete collector;
	}

	if( subscriber )
	{
		delete subscriber;
	}

	if( pusher )
	{
		delete pusher;
	}

	// Clean up the memory used by the map values.
//	for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end();
//		 ++it) {
//		delete it->second;
//	}

}

unsigned PopChannelMap::master_poll()
{
	unsigned updates = 0;

	zmq::pollitem_t items [] = {
			{ *collector, 0, ZMQ_POLLIN, 0 }
	};

	zmq::message_t message;

	do {
		// items, number of items in array, timeout (-1 is block forever)
		zmq::poll (items, 1, 0);

		if (items[0].revents & ZMQ_POLLIN) {
			//  Read message filter
			std::string filter = s_recv(*collector);

			//  Read message contents
			std::string contents = s_recv(*collector);

#ifdef CHANNEL_MAP_VERBOSE
			std::cout << "[" << filter << "] " << contents << std::endl;
#endif

			patch_datastore(contents);

			updates++;
		}
	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

unsigned PopChannelMap::slave_poll()
{
	unsigned updates = 0;

	zmq::pollitem_t items [] = {
			{ *subscriber, 0, ZMQ_POLLIN, 0 }
	};

	zmq::message_t message;

	do {
		// items, number of items in array, timeout (-1 is block forever)
		zmq::poll (items, 1, 0);

		if (items[0].revents & ZMQ_POLLIN) {
			//  Read message filter
			std::string filter = s_recv(*subscriber);

			//  Read message contents
			std::string contents = s_recv(*subscriber);

#ifdef CHANNEL_MAP_VERBOSE
			std::cout << "[" << filter << "] " << contents << std::endl;
#endif

			patch_datastore(contents);

			updates++;
		}
	} while(items[0].revents & ZMQ_POLLIN);

	return updates;
}

// returns number of (valid or invalid) updates / messages received during poll
unsigned PopChannelMap::poll()
{
	if( master )
	{
		return master_poll();
	}
	else
	{
		return slave_poll();
	}
}

// You must hold mutext to call this function
bool PopChannelMap::map_full()
{
	return the_map_.size() >= POP_SLOT_COUNT;
}

uint8_t PopChannelMap::get_update_autoinc()
{
	static uint8_t update = 0;
	return update++;
}


// if a slave has taken an action which requires a roundtrip to the server, this will return true until
// all pending updates have been given by the server
bool PopChannelMap::dirty()
{
	return dirty_;
}

void PopChannelMap::request_sync(void)
{
	if( master )
	{
		sync_table();
	}
	else
	{
		dirty_ = true;
		std::string message = "{\"command\":\"sync\"}";
		s_sendmore (*pusher, "CHANNEL_MAP_SLAVE");
		s_send (*pusher, message);
	}
}

void PopChannelMap::sync_table(void)
{
	mutex::scoped_lock lock(mtx_);

	// this is a master only function
	if( master )
	{
		for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end(); ++it)
		{
			const PopChannelMapKey& key = it->first;
			const PopChannelMapValue& val = it->second;
			set(key, val);
		}
		notify_clean();
	}
}

// tell slaves that for now all updates have been sent
void PopChannelMap::notify_clean()
{
	if( !master )
	{
		return;
	}

	std::string message = "{\"command\":\"set_clean\"}"; // unset_dirty would sound weird
	s_sendmore (*publisher, "CHANNEL_MAP");
	s_send (*publisher, message);
}


void PopChannelMap::set(uint16_t slot, uint64_t tracker, std::string basestation)
{
	mutex::scoped_lock lock(mtx_);

	PopChannelMapKey key;
	key.slot = slot;

	PopChannelMapValue val;
	val.tracker = tracker;
	val.basestation = basestation;

	if( !master )
	{
		dirty_ = true;
	}

	set(key, val);

	if( master )
	{
		notify_clean();
	}
}

// you must hold mutex to call this function
void PopChannelMap::set(PopChannelMapKey key, PopChannelMapValue val)
{
	ostringstream os;
	os << "{\"slot\":" << key.slot << ",\"tracker\":" << val.tracker << ",\"basestation\":\"" << val.basestation << '"'; // json message is missing ending '}'
	string message;

	if( master )
	{
		os << ",\"id\":" << (int)get_update_autoinc() << "}";
		message = os.str();
		s_sendmore (*publisher, "CHANNEL_MAP");
		s_send (*publisher, message);
		the_map_[key] = val;
	}
	else
	{
		os << "}";
		message = os.str();
		s_sendmore (*pusher, "CHANNEL_MAP_SLAVE");
		s_send (*pusher, message);
	}
}

void PopChannelMap::patch_datastore(std::string str)
{
	const char *json = str.c_str();

	struct json_token arr[POP_CHANNEL_MAP_TOKENS];
	const struct json_token *slotTok = 0, *trackerTok = 0, *idTok = 0, *basestationTok = 0, *commandTok = 0, *paramsTok = 0, *p0 = 0, *p1 = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_CHANNEL_MAP_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		// skip printing this message for simple newline messages.  if one string matches, it returns 0 which we then multiply
		if( ( str.compare("\r\n\r\n") * str.compare("\r\n") * str.compare("\n") * str.compare("\r") ) != 0)
		{
			cout << "problem with json string (" <<  str << ")" << endl;
		}
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

	commandTok = find_json_token(arr, "command");
	if( commandTok && commandTok->type == JSON_TYPE_STRING )
	{
		if( FROZEN_GET_STRING(commandTok).compare("sync") == 0 )
		{
			sync_table();
			return;
		}

		if( FROZEN_GET_STRING(commandTok).compare("set_clean") == 0 )
		{
			dirty_ = false;
			return;
		}

		if( FROZEN_GET_STRING(commandTok).compare("request_block") == 0 )
		{
			paramsTok = find_json_token(arr, "params");
			if( paramsTok && paramsTok->type == JSON_TYPE_ARRAY )
			{
				p0 = find_json_token(arr, "params[0]");
				p1 = find_json_token(arr, "params[1]");

				if( p0 && p1 && p0->type == JSON_TYPE_STRING && p1->type == JSON_TYPE_NUMBER)
				{
					std::string bs = FROZEN_GET_STRING(p0);
					unsigned count = parseNumber<unsigned>(FROZEN_GET_STRING(p1));

#ifdef CHANNEL_MAP_VERBOSE
					cout << "Request for block from basestation: " << bs << ", count: " << count << endl;
#endif

					get_block(bs, count);
					return;
				}
			}

		}

	}


	slotTok = find_json_token(arr, "slot");
	if( !(slotTok && slotTok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}

	trackerTok = find_json_token(arr, "tracker");
	if( !(trackerTok && trackerTok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}

	basestationTok = find_json_token(arr, "basestation");
	if( !(basestationTok && basestationTok->type == JSON_TYPE_STRING) )
	{
		return;
	}

	// apply the actual patch
	PopChannelMapKey key;
	key.slot = parseNumber<uint16_t>(FROZEN_GET_STRING(slotTok));

	PopChannelMapValue val;
	val.tracker = parseNumber<uint64_t>(FROZEN_GET_STRING(trackerTok));
	val.basestation = FROZEN_GET_STRING(basestationTok);

	// only need the mutex from here on out
	mutex::scoped_lock lock(mtx_);

	if( master )
	{
		 // set the val, and also tell all slaves
		set(key, val);
		notify_clean();
	}
	else
	{
		// set the val
		the_map_[key] = val;
	}
}

// Slave only, request a block of non sequential time slots
// returns -1 for error, or count of blocks
int32_t PopChannelMap::request_block(unsigned count)
{
	if( master )
	{
		return -1;
	}

	char hostname[256];
	int ret = gethostname(hostname, 256);
	if( ret != 0 )
	{
		cout << "couldn't read linux hostname!" << endl;
		strncpy(hostname, "unkown", 256);
	}

	ostringstream os;
	os << "{\"command\":\"request_block\", \"params\":[\"" << hostname << "\"," << count << "]}";

	dirty_ = true;

	s_sendmore (*pusher, "CHANNEL_MAP_SLAVE");
	s_send (*pusher, os.str());


	return 0;
}

// returns success
bool PopChannelMap::get_block(std::string bs, unsigned count)
{
	if( !master )
	{
		return false;
	}

	mutex::scoped_lock lock(mtx_);

	if( count > (POP_SLOT_COUNT - the_map_.size()) )
	{
		std::cout << "Not enough room in map" << std::endl;
		notify_clean();
		return false;
	}

	unsigned walk = std::max(1u, POP_SLOT_COUNT / count); // floor math
	unsigned i;

	PopChannelMapKey key;
	unsigned given = 0;

	i = 0;

	while(given < count)
	{
		if( map_full() )
		{
			// we shouldn't have entered this function at all
			std::cout << "map filled up unexpectedly" << std::endl;
		}

		PopChannelMapValue val;
		val.tracker = 0; // "0" is a special value which means "assigned to basestation, but not given to a tracker"
		val.basestation = bs;

		key.slot = i;
		if( the_map_.count(key) == 0 || the_map_[key].basestation.compare("") == 0 ) // map key doesn't exist, or if it does, it's empty
		{
			given++;
			set(key, val);
#ifdef CHANNEL_MAP_VERBOSE
			std::cout << "giving out key " << i << std::endl;
#endif
			i = (i + walk)%POP_SLOT_COUNT;
		}
		else
		{
			i++;
		}
	}

	notify_clean();

	return true;
}

bool PopChannelMap::PopChannelMapKeyCompare::operator()(const PopChannelMapKey& a,
												 const PopChannelMapKey& b) const
{
//	if (a.full_secs != b.full_secs)
//		return a.full_secs < b.full_secs;
//	if (a.tracker_id != b.tracker_id)
//		return a.tracker_id < b.tracker_id;
//	if (a.base_station != b.base_station)
//		return a.base_station->hostname() < b.base_station->hostname();

	return a.slot < b.slot;
}

void PopChannelMap::checksum_dump(void)
{
	mutex::scoped_lock lock(mtx_);

	cout << endl;
	ostringstream os;
	for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end(); ++it)
	{
		const PopChannelMapKey& key = it->first;
		const PopChannelMapValue& value = it->second;
		os << key.slot << ", " << value.tracker << ", " << value.basestation << endl;
	}

	cout << os.str();

	uint16_t checksum;

	checksum = crcSlow((uint8_t*)os.str().c_str(), os.str().length());

	cout << endl << "Table Checksum: " << checksum << endl;

	cout << endl;
}

}
