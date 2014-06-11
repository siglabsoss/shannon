/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_CHANNEL_MAP__
#define __POP_CHANNEL_MAP__

#include <stdint.h>
#include <time.h>
#include <zmq.hpp>
#include <iostream>
#include <unistd.h>
#include <sstream>

#include <map>
#include <string>
#include <utility>

#include <boost/thread/mutex.hpp>



namespace pop
{

// Distributed In-memory store of basestation channel map
class PopChannelMap
{
public:
	PopChannelMap(bool, zmq::context_t&);
	~PopChannelMap();


	bool map_full();
	bool get_block(unsigned count);
	unsigned poll();
	void set(uint16_t slot, uint64_t tracker, uint32_t basestation);
	void checksum_dump(void);
	void request_sync(void);
	void sync_table(void);


private:
	struct MapKey
	{
		uint16_t slot;
	};

	struct MapValue
	{
		uint64_t tracker;

		uint32_t basestation;
	};

	void set(MapKey key, MapValue val);
	unsigned master_poll();
	unsigned slave_poll();
	uint8_t get_update_autoinc();
	void patch_datastore(std::string s);


	// Custom less-than comparison function for the map keys.
	struct MapKeyCompare
	{
		bool operator()(const MapKey& a, const MapKey& b) const;
	};

	typedef std::map<MapKey, MapValue, MapKeyCompare> MapType;



	bool master; // this instance is the source of truth?
	zmq::socket_t* publisher;
	zmq::socket_t* subscriber;
	zmq::socket_t* collector;
	zmq::socket_t* pusher; // first time is free

	MapType the_map_;
	mutable boost::mutex mtx_;
};

}

#endif
