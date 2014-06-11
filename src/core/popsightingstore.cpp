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

#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/thread_time.hpp>

#include "core/popbasestation.hpp"
#include "core/popgeolocation.hpp"
#include "core/popsighting.hpp"
#include "core/popsightingstore.hpp"
#include "core/poptrackerlocationstore.hpp"

using boost::bind;
using boost::get_system_time;
using boost::mutex;
using boost::posix_time::milliseconds;
using boost::system_time;
using boost::thread;
using std::make_pair;
using std::pair;
using std::string;
using std::tr1::unordered_map;
using std::vector;

namespace pop
{

// Number of seconds to wait before aggregating sighting data. The delay exists
// to allow for network latency between the base stations and the S3P.
const int PopSightingStore::PROCESSING_DELAY_SEC = 5;

// Size of the time window, in milliseconds, that's used to group sightings from
// the same tracker. Since light travels about 300 km in a millisecond,
// multiplying this number by 300 km will yield the range of possible tracker
// locations.
const int PopSightingStore::AGGREGATION_WINDOW_MSEC = 100;

namespace
{

int time_diff_msec(time_t a_full_secs, double a_frac_secs,
				   time_t b_full_secs, double b_frac_secs)
{
	return static_cast<int>(a_full_secs - b_full_secs) * 1000 +
		static_cast<int>((a_frac_secs - b_frac_secs) * 1000.0);
}

}  // namespace

PopSightingStore::PopSightingStore(
	const PopGeoLocation* geo_location,
	PopTrackerLocationStore* tracker_location_store)
	: geo_location_(geo_location),
	  tracker_location_store_(tracker_location_store),
	  stopping_(false),
	  aggregation_thread_(NULL)
{
	assert(geo_location != NULL);
	assert(tracker_location_store != NULL);
}

PopSightingStore::~PopSightingStore()
{
	{
		mutex::scoped_lock lock(mtx_);

		// Clean up the memory used by the map values.
		for (MapType::const_iterator it = the_map_.begin();
			 it != the_map_.end(); ++it) {
			delete it->second;
		}

		// Clean up the memory used by the PopBaseStation objects.
		for (unordered_map<string, PopBaseStation*>::const_iterator it =
				 base_stations_.begin();
			 it != base_stations_.end(); ++it) {
			delete it->second;
		}
	}

	// Clean up the memory used by the boost::thread object.
	if (aggregation_thread_ != NULL)
		delete aggregation_thread_;
}

void PopSightingStore::start_thread()
{
	assert(aggregation_thread_ == NULL);

	aggregation_thread_ = new thread(
		bind(&PopSightingStore::aggregate_sightings, this));
}

void PopSightingStore::stop_thread()
{
	assert(aggregation_thread_ != NULL);

	{
		mutex::scoped_lock lock(mtx_);
		assert(!stopping_);
		stopping_ = true;
		stopping_cond_.notify_all();
	}

	aggregation_thread_->join();
}

void PopSightingStore::add_sighting(const PopSighting& sighting)
{
	MapKey key;
	key.full_secs = sighting.full_secs;
	key.frac_secs = sighting.frac_secs;

	MapValue* const value = new MapValue();
	value->base_station = get_base_station(sighting.hostname);
	value->tracker_id = sighting.tracker_id;
	value->lat = sighting.lat;
	value->lng = sighting.lng;

	{
		mutex::scoped_lock lock(mtx_);
		the_map_.insert(make_pair(key, value));
	}
}

bool PopSightingStore::MapKeyCompare::operator()(const MapKey& a,
												 const MapKey& b) const
{
	return map_key_less_than(a, b);
}

// This is the main method for the aggregation thread.
void PopSightingStore::aggregate_sightings()
{
	MapKey current_map_key;
	current_map_key.full_secs = 0;
	current_map_key.frac_secs = 0.0;

	mutex::scoped_lock lock(mtx_);

	for (;;) {
		unordered_map<uint64_t, vector<PopSighting> > tracker_sightings;
		scan_sighting_map(&current_map_key, &tracker_sightings);

		for (unordered_map<uint64_t, vector<PopSighting> >::iterator it =
				 tracker_sightings.begin();
			 it != tracker_sightings.end(); ++it) {
			flush_sighting_vector(&it->second);
		}

		const system_time timeout = get_system_time() + milliseconds(500);
		if (stopping_cond_.timed_wait(
				lock, timeout, bind(&PopSightingStore::is_stopping, this))) {
		    return;
		}
	}
}

void PopSightingStore::scan_sighting_map(
	MapKey* current_map_key,
	unordered_map<uint64_t, vector<PopSighting> >* tracker_sightings)
{
	assert(current_map_key != NULL);
	assert(tracker_sightings != NULL);

	const time_t now = time(NULL);

	for (;;) {
		if (the_map_.empty())
			return;

		const MapType::iterator it = the_map_.begin();
		const MapKey& map_key = it->first;
		MapValue* const map_value = it->second;

		if (map_key.full_secs >= now - PROCESSING_DELAY_SEC)
			return;

		if (!map_key_less_than(map_key, *current_map_key)) {
			*current_map_key = map_key;

			vector<PopSighting>* const sightings =
				&(*tracker_sightings)[map_value->tracker_id];

			if (!sightings->empty()) {
				const PopSighting& first_sighting = sightings->front();

				if (time_diff_msec(
						map_key.full_secs, map_key.frac_secs,
						first_sighting.full_secs, first_sighting.frac_secs) >
					AGGREGATION_WINDOW_MSEC) {
					flush_sighting_vector(sightings);
				}

				sightings->resize(sightings->size() + 1);
				PopSighting* const new_sighting = &sightings->back();

				new_sighting->hostname = map_value->base_station->hostname();
				new_sighting->tracker_id = map_value->tracker_id;
				new_sighting->lat = map_value->lat;
				new_sighting->lng = map_value->lng;
				new_sighting->full_secs = map_key.full_secs;
				new_sighting->frac_secs = map_key.frac_secs;
			}
		}

		delete map_value;
		the_map_.erase(it);
	}
}

void PopSightingStore::flush_sighting_vector(vector<PopSighting>* sightings)
{
	assert(sightings != NULL);

	if (sightings->empty())
		return;

	const PopSighting& first_sighting = sightings->front();

	double lat = 0.0;
	double lng = 0.0;
	if (geo_location_->calculate_location(*sightings, &lat, &lng)) {
		tracker_location_store_->report_tracker_location(
			first_sighting.tracker_id, first_sighting.full_secs, lat, lng);
	}

	sightings->clear();
}

// Returns a unique base station pointer for the given hostname.
const PopBaseStation* PopSightingStore::get_base_station(const string& hostname)
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

// static
bool PopSightingStore::map_key_less_than(const MapKey& a, const MapKey& b)
{
	if (a.full_secs != b.full_secs)
		return a.full_secs < b.full_secs;

	return a.frac_secs < b.frac_secs;
}

}
