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

//#define CHANNEL_MAP_VERBOSE

namespace pop
{

// FIXME: copy the logic of http://zguide.zeromq.org/php:chapter5#Reliable-Pub-Sub-Clone-Pattern

PopChannelMap::PopChannelMap(bool m, zmq::context_t& context) : master(m), publisher(0), subscriber(0)
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
		subscriber->connect("tcp://localhost:11526");
		subscriber->setsockopt( ZMQ_SUBSCRIBE, "CHANNEL_MAP", 11);
		pusher = new zmq::socket_t(context, ZMQ_PUSH);
		pusher->connect("tcp://localhost:11527");

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

void PopChannelMap::request_sync(void)
{
	if( master )
	{
		sync_table();
	}
	else
	{
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
			const MapKey& key = it->first;
			const MapValue& val = it->second;
			set(key, val);
		}
	}
}


void PopChannelMap::set(uint16_t slot, uint64_t tracker, uint32_t basestation)
{
	mutex::scoped_lock lock(mtx_);

	MapKey key;
	key.slot = slot;

	MapValue val;
	val.tracker = tracker;
	val.basestation = basestation;

	set(key, val);
}

// you must hold mutex to call this function
void PopChannelMap::set(MapKey key, MapValue val)
{
	ostringstream os;
	os << "{\"slot\":" << key.slot << ",\"tracker\":" << val.tracker << ",\"basestation\":" << val.basestation; // json message is missing ending '}'
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
	const struct json_token *slotTok = 0, *trackerTok = 0, *idTok = 0, *basestationTok = 0, *commandTok;

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
	if( !(basestationTok && basestationTok->type == JSON_TYPE_NUMBER) )
	{
		return;
	}

	// apply the actual patch
	MapKey key;
	key.slot = parseNumber<uint16_t>(FROZEN_GET_STRING(slotTok));

	MapValue val;
	val.tracker = parseNumber<uint64_t>(FROZEN_GET_STRING(trackerTok));
	val.basestation = parseNumber<uint64_t>(FROZEN_GET_STRING(basestationTok));

	// only need the mutex from here on out
	mutex::scoped_lock lock(mtx_);

	if( master )
	{
		 // set the val, and also tell all slaves
		set(key, val);
	}
	else
	{
		// set the val
		the_map_[key] = val;
	}
}


// returns success
bool PopChannelMap::get_block(unsigned count)
{
	mutex::scoped_lock lock(mtx_);

	if( count > (POP_SLOT_COUNT - the_map_.size()) )
	{
		std::cout << "Not enough room in map" << std::endl;
		return false;
	}

	unsigned walk = std::max(1u, POP_SLOT_COUNT / count); // floor math
	unsigned i;

	MapKey key;
	unsigned given = 0;

	i = 0;

	while(given < count)
	{
		if( map_full() )
		{
			// we shouldn't have entered this function at all
			std::cout << "map filled up unexpectedly" << std::endl;
		}

		MapValue val;
		val.tracker = 34;
		val.basestation = 342;

		key.slot = i;
		if( the_map_.count(key) == 0 ) //if( the_map_.find(key) == the_map_.end() )
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

	return true;
}

bool PopChannelMap::MapKeyCompare::operator()(const MapKey& a,
												 const MapKey& b) const
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
		const MapKey& key = it->first;
		const MapValue& value = it->second;
		os << key.slot << ", " << value.tracker << ", " << value.basestation << endl;
	}

	cout << os.str();

	uint16_t checksum;

	checksum = crcSlow((uint8_t*)os.str().c_str(), os.str().length());

	cout << endl << "Table Checksum: " << checksum << endl;

	cout << endl;
}

}
