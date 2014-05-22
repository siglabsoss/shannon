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
#include <utility>
#include <vector>

#include "core/popmultilateration.hpp"
#include "core/popsighting.hpp"
#include "core/popsightingstore.hpp"
#include "core/poptrackerlocationstore.hpp"

using std::make_pair;
using std::pair;
using std::vector;

namespace pop
{

const int PopSightingStore::MIN_NUM_BASESTATIONS = 3;

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
	// Clean up the memory used by the map values.
	for (MapType::const_iterator it = the_map_.begin(); it != the_map_.end();
		 ++it) {
		delete it->second;
	}
}

void PopSightingStore::add_sighting(const PopSighting& sighting)
{
	MapKey key;
	key.full_secs = sighting.full_secs;
	key.tracker_id = sighting.tracker_id;
	key.hostname = sighting.hostname;

	MapValue* const value = new MapValue();
	value->lat = sighting.lat;
	value->lng = sighting.lng;
	value->frac_secs = sighting.frac_secs;

	// If multiple sightings are received with the same (full_secs, serial,
	// hostname) key, only the first sighting will be recorded.
	// TODO(snyderek): Is this the desired behavior?
	if (!the_map_.insert(make_pair(key, value)).second) {
		// The key already exists in the map. Delete the newly allocated value
		// to avoid a memory leak.
		delete value;
	}

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
	if (a.hostname != b.hostname)
		return a.hostname < b.hostname;

	return false;
}

// Builds a vector of all sightings for the given time and tracker. If there are
// at least three sightings, performs multilateration and stores the computed
// tracker location.
void PopSightingStore::aggregate_sightings(time_t full_secs,
										   uint64_t tracker_id)
{
	vector<PopSighting> sightings;

	const pair<MapType::const_iterator, MapType::const_iterator> range =
		get_sighting_range(full_secs, tracker_id);

	for (MapType::const_iterator it = range.first; it != range.second; ++it) {
		const MapKey& key = it->first;
		const MapValue& value = *it->second;

		assert(key.full_secs == full_secs);
		assert(key.tracker_id == tracker_id);

		sightings.resize(sightings.size() + 1);
		PopSighting* const sighting = &sightings.back();

		sighting->hostname = key.hostname;
		sighting->tracker_id = tracker_id;
		sighting->lat = value.lat;
		sighting->lng = value.lng;
		sighting->full_secs = full_secs;
		sighting->frac_secs = value.frac_secs;
	}

	if (sightings.size() >=
		static_cast<vector<PopSighting>::size_type>(MIN_NUM_BASESTATIONS)) {
		double lat = 0.0;
		double lng = 0.0;
		multilateration_->calculate_location(sightings, &lat, &lng);

		tracker_location_store_->report_device_location(tracker_id, full_secs,
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
pair<PopSightingStore::MapType::const_iterator,
	 PopSightingStore::MapType::const_iterator>
PopSightingStore::get_sighting_range(time_t full_secs,
									 uint64_t tracker_id) const
{
	pair<MapType::const_iterator, MapType::const_iterator> range;

	MapKey key;
	key.full_secs = full_secs;
	key.tracker_id = tracker_id;

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

}
