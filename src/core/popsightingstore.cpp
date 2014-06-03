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
#include <tr1/unordered_map>
#include <utility>
#include <vector>

#include <boost/thread/mutex.hpp>

#include "core/popbasestation.hpp"
#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"
#include "core/popsightingstore.hpp"
#include "core/poptrackerlocationstore.hpp"

using boost::mutex;
using std::make_pair;
using std::pair;
using std::string;
using std::tr1::unordered_map;
using std::vector;

namespace pop
{

PopSightingStore::PopSightingStore(
	const PopMultilateration* multilateration,
	PopTrackerLocationStore* tracker_location_store)
	: multilateration_(multilateration),
	  tracker_location_store_(tracker_location_store)
{
	assert(multilateration != NULL);
	assert(tracker_location_store != NULL);
}

PopSightingStore::~PopSightingStore()
{
	mutex::scoped_lock lock(mtx_);

	// Clean up the memory used by the map values.
	for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end();
		 ++it) {
		delete it->second;
	}

	// Clean up the memory used by the PopBaseStation objects.
	for (unordered_map<string, PopBaseStation*>::const_iterator it =
			 base_stations_.begin();
		 it != base_stations_.end(); ++it) {
		delete it->second;
	}
}

void PopSightingStore::add_sighting(const PopSighting& sighting)
{
	MapKey key;
	key.full_secs = sighting.full_secs;
	key.tracker_id = sighting.tracker_id;
	key.base_station = GetBaseStation(sighting.hostname);

	MapValue* const value = new MapValue();
	value->lat = sighting.lat;
	value->lng = sighting.lng;
	value->frac_secs = sighting.frac_secs;

	bool inserted = false;
	{
		mutex::scoped_lock lock(mtx_);

		// If multiple sightings are received with the same (full_secs, serial,
		// base_station) key, only the first sighting will be recorded.
		// TODO(snyderek): Is this the desired behavior?
		inserted = the_map_.insert(make_pair(key, value)).second;
	}

	// If the key already exists in the map, delete the newly allocated map
	// value to avoid a memory leak.
	if (!inserted)
		delete value;

	aggregate_sightings(sighting.full_secs, sighting.tracker_id);

	// TODO(snyderek): Delete old sightings.
}

bool PopSightingStore::MapKeyCompare::operator()(const MapKey& a,
												 const MapKey& b) const
{
	if (a.full_secs != b.full_secs)
		return a.full_secs < b.full_secs;
	if (a.tracker_id != b.tracker_id)
		return a.tracker_id < b.tracker_id;
	if (a.base_station != b.base_station) {
		return b.base_station != NULL &&
			   (a.base_station == NULL ||
			    a.base_station->hostname() < b.base_station->hostname());
	}

	return false;
}

// Builds a vector of all sightings for the given time and tracker. If the
// number of sightings is at least PopMultilateration::MIN_NUM_BASESTATIONS,
// performs multilateration and stores the computed tracker location.
void PopSightingStore::aggregate_sightings(time_t full_secs,
										   uint64_t tracker_id)
{
	vector<PopSighting> sightings;

	{
		mutex::scoped_lock lock(mtx_);

		const pair<MapType::const_iterator, MapType::const_iterator> range =
			get_sighting_range(full_secs, tracker_id);

		for (MapType::const_iterator it = range.first; it != range.second;
			 ++it) {
			const MapKey& key = it->first;
			const MapValue& value = *it->second;

			assert(key.full_secs == full_secs);
			assert(key.tracker_id == tracker_id);

			sightings.resize(sightings.size() + 1);
			PopSighting* const sighting = &sightings.back();

			sighting->hostname = key.base_station->hostname();
			sighting->tracker_id = tracker_id;
			sighting->lat = value.lat;
			sighting->lng = value.lng;
			sighting->full_secs = full_secs;
			sighting->frac_secs = value.frac_secs;
		}
	}

	if (sightings.size() >=
		static_cast<vector<PopSighting>::size_type>(
			PopMultilateration::MIN_NUM_BASESTATIONS)) {
		double lat = 0.0;
		double lng = 0.0;
		multilateration_->calculate_location(sightings, &lat, &lng);

		tracker_location_store_->report_tracker_location(tracker_id, full_secs,
														 lat, lng);
	}
}

// Returns a (begin, end) iterator pair for the range of map entries that match
// the given (full_secs, tracker_id) partial key. You can use this range to
// iterate over the subset of key-value pairs in 'the_map_'.
//
// The first iterator points to the first map entry that is greater than or
// equal to the partial key. The second iterator points to the first map entry
// that is strictly greater than the partial key [or end() if no such key
// exists]. If no matching entries are found, the two returned iterators will be
// equal.
//
// mtx_ must be locked when this function is called.
pair<PopSightingStore::MapType::const_iterator,
	 PopSightingStore::MapType::const_iterator>
PopSightingStore::get_sighting_range(time_t full_secs,
									 uint64_t tracker_id) const
{
	pair<MapType::const_iterator, MapType::const_iterator> range;

	MapKey key;
	key.full_secs = full_secs;
	key.tracker_id = tracker_id;
	key.base_station = NULL;

	range.first = the_map_.lower_bound(key);

	++key.tracker_id;
	// Check for integer overflow.
	if (key.tracker_id > 0) {
		// Normal case.
		range.second = the_map_.lower_bound(key);
	} else {
		// tracker_id was already the maximum uint64_t value. Use the_map_.end()
		// as the end of the range.
		range.second = the_map_.end();
	}

	return range;
}

// Returns a unique base station pointer for the given hostname.
const PopBaseStation* PopSightingStore::GetBaseStation(const string& hostname)
{
	assert(!hostname.empty());

	mutex::scoped_lock lock(mtx_);

	const pair<unordered_map<string, PopBaseStation*>::iterator, bool>
		insert_result = base_stations_.insert(
			pair<string, PopBaseStation*>(hostname, NULL));

	PopBaseStation** const base_station = &insert_result.first->second;

	if (insert_result.second)
		*base_station = new PopBaseStation(hostname);

	return *base_station;
}

}
