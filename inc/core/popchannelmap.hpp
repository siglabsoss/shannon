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

//class PopMultilateration;
//class PopTrackerLocationStore;
//struct PopSighting;

// In-memory store of tracker sightings reported by base stations.
class PopChannelMap
{
public:
	PopChannelMap(bool, zmq::context_t&);
	~PopChannelMap();

//	void add_sighting(const PopSighting& sighting);

	bool map_full();
	bool get_block(unsigned count);
	void poll();



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

	// Custom less-than comparison function for the map keys.
	struct MapKeyCompare
	{
		bool operator()(const MapKey& a, const MapKey& b) const;
	};

	typedef std::map<MapKey, MapValue, MapKeyCompare> MapType;

//	void aggregate_sightings(time_t full_secs, uint64_t tracker_id);
//	std::pair<MapType::const_iterator, MapType::const_iterator>
//		get_sighting_range(time_t full_secs, uint64_t tracker_id) const;


	bool master; // this instance is the source of truth?
	zmq::socket_t* publisher;
	zmq::socket_t* subscriber;

	MapType the_map_;
//	std::tr1::unordered_map<std::string, PopBaseStation*> base_stations_;
	mutable boost::mutex mtx_;
};

}

#endif
