/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_SIGHTING_STORE__
#define __POP_SIGHTING_STORE__

#include <stdint.h>
#include <time.h>

#include <map>
#include <string>
#include <tr1/unordered_map>
#include <utility>

#include <boost/thread/mutex.hpp>

namespace pop
{

class PopBaseStation;
class PopGeoLocation;
class PopTrackerLocationStore;
struct PopSighting;

// In-memory store of tracker sightings reported by base stations.
class PopSightingStore
{
public:
	PopSightingStore(const PopGeoLocation* geo_location,
					 PopTrackerLocationStore* tracker_location_store);
	~PopSightingStore();

	void add_sighting(const PopSighting& sighting);

private:
	struct MapKey
	{
		// Integer component of the timestamp when the tracker signal was
		// received by the base station. (Seconds since the epoch.)
		time_t full_secs;

		// Unique ID from the tracker board.
		// TODO(snyderek): What is the correct data type?
		uint64_t tracker_id;

		// Base station that received the tracker signal.
		const PopBaseStation* base_station;
	};

	struct MapValue
	{
		// Latitude and longitude
		double lat;
		double lng;

		// Fractional component of the timestamp when the tracker signal was
		// received by the base station.
		double frac_secs;
	};

	// Custom less-than comparison function for the map keys.
	struct MapKeyCompare
	{
		bool operator()(const MapKey& a, const MapKey& b) const;
	};

	typedef std::map<MapKey, MapValue*, MapKeyCompare> MapType;

	void aggregate_sightings(time_t full_secs, uint64_t tracker_id);
	std::pair<MapType::const_iterator, MapType::const_iterator>
		get_sighting_range(time_t full_secs, uint64_t tracker_id) const;

	const PopBaseStation* GetBaseStation(const std::string& hostname);

	const PopGeoLocation* const geo_location_;
	PopTrackerLocationStore* const tracker_location_store_;

	MapType the_map_;
	std::tr1::unordered_map<std::string, PopBaseStation*> base_stations_;
	mutable boost::mutex mtx_;
};

}

#endif
