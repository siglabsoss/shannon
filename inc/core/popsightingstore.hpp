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
#include <vector>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include "core/popsighting.hpp"

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

	// Starts and stops the aggregation method. start_thread() must be called
	// before stop_thread(). Each method can be called at most once.
	void start_thread();
	void stop_thread();

	void add_sighting(const PopSighting& sighting);

private:
	static const int PROCESSING_DELAY_SEC;
	static const int AGGREGATION_WINDOW_MSEC;

	struct MapKey
	{
		// Integer component of the timestamp when the tracker signal was
		// received by the base station. (Seconds since the epoch.)
		time_t full_secs;

		// Fractional component of the timestamp when the tracker signal was
		// received by the base station.
		double frac_secs;
	};

	struct MapValue
	{
		// Base station that received the tracker signal.
		const PopBaseStation* base_station;

		// Unique ID from the tracker board.
		// TODO(snyderek): What is the correct data type?
		uint64_t tracker_id;

		// Latitude and longitude
		double lat;
		double lng;
	};

	// Custom less-than comparison function for the map keys.
	struct MapKeyCompare
	{
		bool operator()(const MapKey& a, const MapKey& b) const;
	};

	typedef std::multimap<MapKey, MapValue*, MapKeyCompare> MapType;

	bool is_stopping() const { return stopping_; }

	void aggregate_sightings();
	void scan_sighting_map(
		MapKey* current_map_key,
		std::tr1::unordered_map<uint64_t, std::vector<PopSighting> >*
			tracker_sightings);
	void flush_sighting_vector(std::vector<PopSighting>* sightings);

	const PopBaseStation* get_base_station(const std::string& hostname);

	static bool map_key_less_than(const MapKey& a, const MapKey& b);

	const PopGeoLocation* const geo_location_;
	PopTrackerLocationStore* const tracker_location_store_;

	MapType the_map_;
	std::tr1::unordered_map<std::string, PopBaseStation*> base_stations_;
	bool stopping_;
	mutable boost::condition_variable stopping_cond_;
	mutable boost::mutex mtx_;

	boost::thread* aggregation_thread_;
};

}

#endif
