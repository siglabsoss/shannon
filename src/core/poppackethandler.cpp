#include <iostream>
#include <boost/timer.hpp>
#include <algorithm>    // std::sort

#include "core/poppackethandler.hpp"
#include "core/util.h"
#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"
#include "core/utilities.hpp"
#include "core/popchannelmap.hpp"
#include "dsp/prota/popsparsecorrelate.h"
#include "b64/b64.h"


using namespace std;

namespace pop
{



#define QUICK_SEARCH_STEPS (1296)
#define DATA_SAMPLE(x) data[x]

// how good of a match is required to attempt demodulate
#define COMB_COORELATION_FACTOR ((double)0.30)


uint32_t pop_correlate_spool(const uint32_t* data, const uint16_t dataSize, const uint32_t* comb, const uint32_t combSize, int32_t* scoreOut, uint32_t* finalSample)
{
	uint32_t denseCombLength = comb[combSize-1] - comb[0];
	uint32_t denseDataLength = 0;

	// the best score possible (100% correlation) is equal to the length of the comb
	uint32_t threshold = COMB_COORELATION_FACTOR * denseCombLength;

	uint32_t i;

	// we are forced to scan through the input data to determine if any modulus events have occurred in order to get a real value for denseDataLength
	for(i = 1; i < dataSize; i++)
	{
		if( DATA_SAMPLE(i) < DATA_SAMPLE(i-1) )
		{
			//denseDataLength += ARTEMIS_CLOCK_SPEED_HZ;
//			printf("bump (%d)\r\n", i);
		}

		denseDataLength += DATA_SAMPLE(i)-DATA_SAMPLE(i-1);
	}

	uint32_t end_padding = 200;

	if( denseDataLength < (denseCombLength+end_padding) )
	{
//		printf("dense data size %"PRIu32" must not be less than dense comb size %"PRIu32"\r\n", denseDataLength, denseCombLength);

		*scoreOut = 0;

		//FIXME: this is not an appropriate way of returning an error condition
		return 0;
	}

	int32_t score, scoreLeft, scoreRight, maxScoreQuick = 0, maxScore = 0; //x(key)score
	uint32_t maxScoreOffsetQuick, maxScoreOffset, scoreOffsetBinSearch, iterations, combOffset;


	// Artemis is given a "guess" of the start timer value when the start-of-frame should occur
#ifdef POPWI_PLATFORM_ARTEMIS
	iterations = guess - DATA_SAMPLE(0) + GUESS_ERROR + 1;
	combOffset = guess - DATA_SAMPLE(0) - GUESS_ERROR;
#else
	iterations = denseDataLength - denseCombLength + 1;
	combOffset = 0;
#endif

	std::vector<uint32_t> matchOffsets;
	std::vector<int32_t> matchScores;

	int64_t lastOffset = -1;

	int count = 0;

	uint32_t state = 0;

	// quick search
	for(; combOffset < iterations; combOffset += QUICK_SEARCH_STEPS)
	{
		score = do_comb(data, dataSize, comb, combSize, combOffset, &state);

		// if the score passes the threshold
		if( abs(score) > threshold )
		{
			// if we found two consecutive matches (this must be the same packet)
			if( lastOffset != -1 )
			{
				// if this match is better than the last one
				if( abs(score) > abs(matchScores.back()) )
				{
					// remove previous score, and use this one
					matchOffsets.pop_back();
					matchScores.pop_back();
					matchOffsets.push_back(combOffset);
					matchScores.push_back(score);
				}
			}
			else
			{
				// this is the first match above the threshold in awhile, save it
				matchOffsets.push_back(combOffset);
				matchScores.push_back(score);
			}


			// remember the offset of the last match
			lastOffset = combOffset;
		}
		count++;
	}

	// this is the dense value of the last "quick" search we tried
	uint32_t finalDense = combOffset-QUICK_SEARCH_STEPS+data[0];

	uint32_t samp=0;

	for( i = 0; i < dataSize; i++ )
	{
		if( data[i] < finalDense )
		{
			samp = i;
		}
	}

	*finalSample = samp;



//	cout << "got " << matchOffsets.size() << " thresholded combs (" << count << ")" << endl;

	uint32_t ret = 0;

	if( matchOffsets.size() > 1 )
	{
		cout << "got MULTIPLE different packets" << endl;
	}

	for(i = 0; i < matchOffsets.size(); ++i)
	{

//		cout << "climbing match " << i <<  " offset: " << matchOffsets[i] << " score: " << abs(matchScores[i]) << endl;
//		cout << "thresh: " << threshold << endl;




		// we've found a peak
		uint32_t searchStep = QUICK_SEARCH_STEPS;

		maxScoreOffset = scoreOffsetBinSearch = matchOffsets[i];
		maxScore = matchScores[i];


		// warmup loop; we only need to do a single comb because the previous one was done in the quick search
		scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1, &state);

		if( abs(maxScoreQuick) > abs(scoreRight) )
		{
			scoreOffsetBinSearch -= searchStep/2;
		}
		else
		{
			scoreOffsetBinSearch += searchStep/2;
		}


		while( searchStep != 1 )
		{
			searchStep /= 2;

			scoreLeft = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch, &state);

			scoreRight = do_comb(data, dataSize, comb, combSize, scoreOffsetBinSearch+1, &state);

			if( abs(scoreLeft) > abs(scoreRight) )
			{
				if( abs(scoreLeft) > abs(maxScore) )
				{
					maxScore = scoreLeft;
					maxScoreOffset = scoreOffsetBinSearch;
				}

				scoreOffsetBinSearch -= searchStep/2;
			}
			else
			{
				if( abs(scoreRight) > abs(maxScore) )
				{
					maxScore = scoreRight;
					maxScoreOffset = scoreOffsetBinSearch+1;
				}

				scoreOffsetBinSearch += searchStep/2;
			}

			if( searchStep == 1 && scoreLeft == scoreRight )
			{
				//FIXME: this condition can be fixed by curve fitting the searched spots
				printf("Flat peak detected, start of frame will be slightly wrong\r\n");
			}
		}




		printf("max scoreee: %d\r\n", maxScore);


		if( i == 0 )
		{
			//FIXME
			ret = DATA_SAMPLE(0) + maxScoreOffset;

			*scoreOut = maxScore;

		}





	}


	return ret;


//	printf("max: %u %d\r\n", maxScoreOffsetQuick, maxScoreQuick);



//	printf("Max offset bin:   %u\r\n", maxScoreOffset);

	//*scoreOut = maxScore;

//	return DATA_SAMPLE(0) + maxScoreOffset;
}















PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint32_t>("PopPacketHandler", 1500), rpc(0), new_timers(0), artemis_tpm_start(-1)
{

}

// sort by uuid first, then by time
bool PopPacketQueueCompare(const PopPacketHandler::PopPacketQueue& lhs, const PopPacketHandler::PopPacketQueue& rhs)
{
	int comparison;

	comparison = lhs.uuid.compare(rhs.uuid);

	if( comparison < 0 )
	{
		return true;
	}
	else if( comparison > 0 )
	{
		return false;
	}
	else
	{
		return lhs.time.get_real_secs() < rhs.time.get_real_secs();
	}
}

void PopPacketHandler::enqueue_packet(std::string to, ota_packet_t& packet)
{
	cout << "enqueued packet to " << to << endl;
	PopPacketQueue q;

	q.time = get_microsec_system_time();
	q.uuid = to;
	q.packet = packet;

	queue.push_back(q);

	// this is very expensive but fun!
	std::sort (queue.begin(), queue.end(), PopPacketQueueCompare);


//	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
//	{
//		std::cout << ' ' << it->time << ' ' << it->uuid << ' ' << it->packet.data << endl;
//	}
}

// Next packet to be transmitted (requires queue to already be sorted)
// Returns null if no packets are pending
ota_packet_t* PopPacketHandler::peek_packet(std::string uuid)
{
	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
	{
		if( uuid.compare(it->uuid) == 0 )
		{
			return &(it->packet);
		}
//		std::cout << ' ' << it->time << ' ' << it->uuid << ' ' << it->packet.data << endl;
	}

	// no packets waiting
	return NULL;
}

// Deletes a packet from the queue
void PopPacketHandler::erase_packet(std::string uuid, ota_packet_t& packet)
{
	for (std::vector<PopPacketQueue>::iterator it = queue.begin(); it!=queue.end(); ++it)
	{
		if( uuid.compare(it->uuid) == 0 && memcmp(&packet, &(it->packet), sizeof(ota_packet_t) ) == 0)
		{
			queue.erase(it);
			return;
		}
	}
}

// returns the nearest slot for a tracker
int32_t PopPacketHandler::pop_get_nearest_slot(uuid_t uuid, int32_t slot_in)
{
	std::vector<PopChannelMap::PopChannelMapKey> keys;
	std::vector<PopChannelMap::PopChannelMapValue> values;
	map->find_by_tracker(uuid, keys, values);

	int32_t diff = POP_SLOT_COUNT + 1; // worst case
	uint32_t diff_slot = 0;

	int32_t tmp;


	//	cout << "System Slot: " << system_now_slot << endl;
	//	cout << "Closest slot: " << pop_get_tracker_slot_now(uuid) << endl;

	for( unsigned i = 0; i < keys.size(); i++ )
	{
		const PopChannelMap::PopChannelMapKey& key = keys[i];
		PopChannelMap::PopChannelMapValue val = values[i];

		tmp = abs((int32_t) key.slot - slot_in);

		if( tmp < diff )
		{
			diff = tmp;
			diff_slot = key.slot;
		}

		// now compare against wrapped version
		tmp = abs((int32_t) key.slot + POP_SLOT_COUNT - slot_in);

		if( tmp < diff )
		{
			diff = tmp;
			diff_slot = key.slot;
		}
	}

	if( diff == POP_SLOT_COUNT + 1 )
	{
		cout << "something seriously wrong pop_get_tracker_slot_now (was s3p restarted?)" << endl;
		return 0;
	}

	cout << "Slot: " << diff_slot << " is closest to system's now slot: " << slot_in << endl;

	return diff_slot;
}

// takes the "now" timeslot from the system clock and looks through the the timeslot list for device to find the closest slot
// this should be the timeslot that the tracker thinks it is transmitting on
int32_t PopPacketHandler::pop_get_tracker_slot_now(uuid_t uuid)
{
	// according to basestation clock, what slot is it?
	PopTimestamp system_now = get_microsec_system_time();
	uint64_t system_pit = round(system_now.get_frac_secs() * 19200000.0) + system_now.get_full_secs()*19200000;
	int32_t system_now_slot = pop_get_slot_pit_rounded(system_pit);

	return pop_get_nearest_slot(uuid, system_now_slot);
}

int PopPacketHandler::basestation_should_respond(uuid_t uuid)
{
	std::vector<PopChannelMap::PopChannelMapKey> keys;
	std::vector<PopChannelMap::PopChannelMapValue> values;
	map->get_full_map(keys, values);

	std::vector<std::string> basestations;

	// build vector of unique basestation names
	for( unsigned i = 0; i < keys.size(); i++ )
	{
		PopChannelMap::PopChannelMapValue val = values[i];
		if( std::find(basestations.begin(), basestations.end(), val.basestation) == basestations.end() )
		{
			basestations.push_back(val.basestation);
		}
	}

	uint16_t count = basestations.size();
	uint16_t crc = crcSlow(uuid.bytes, sizeof(uuid.bytes)) >> 3; // crcSlow doesn't seem to ever give odd results?

//	cout << "crc " << crc <<  endl;

	// modulus crc of device serial by count of basestations
	uint16_t result = crc % count;

	cout << "crc " << crc << " result " << result << " bs is " << basestations[result] << endl;

	// determine if this basestation should reply to the packet
	if( basestations[result].compare(pop_get_hostname()) == 0 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void PopPacketHandler::execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], char *str, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart)
{
	std::string method = FROZEN_GET_STRING(methodTok);
	const struct json_token *params, *p0, *p1, *p2;

	int32_t original_id = -1;

	if( idTok )
	{
		original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
	}


	if( method.compare("log") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
//			rcp_log(FROZEN_GET_STRING(p0));
//			respond_int(0, methodId);
		}
	}

	if( method.compare("utc_rq") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			uuid_t uuid = b64_to_uuid(FROZEN_GET_STRING(p0));
			cout << "Serial: " << FROZEN_GET_STRING(p0) << endl;


			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			// Traditionally all ota packets are replied with a "tx" rpc here.
			// but in this case we want the basestation to behave smart, so we actually send an rpc

			int32_t original_id = -1;

			if( idTok )
			{
				original_id = parseNumber<int32_t>(FROZEN_GET_STRING(idTok));
			}

			char buf[128];

			// we encapsulate the original rpc id so that the basestation can correctly reply
			snprintf(buf, 128, "{\"method\":\"bs_send_utc_reply\",\"params\":[%d, %d, %ld]}", original_id, txTime, pitTxTime);

			printf("\r\n");
			puts(buf);

			rpc->send_rpc(buf, strlen(buf));
		}
	}

	if( method.compare("slot_rq") == 0  && original_id != -1 )
	{
		p0 = find_json_token(arr, "params[0]");
		if( p0 && p0->type == JSON_TYPE_STRING )
		{
			uuid_t uuid = b64_to_uuid(FROZEN_GET_STRING(p0));

			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			//cout << "slot rq from " << FROZEN_GET_STRING(p0) << endl;

			rpc->fabric->add_name(FROZEN_GET_STRING(p0));



			// how many slots are we giving out?
			unsigned remaining = 5;
			unsigned chosen = 0;
			uint16_t slots[remaining];

			// grab all slots available to us
			std::vector<PopChannelMap::PopChannelMapKey> keys;
			std::vector<PopChannelMap::PopChannelMapValue> values;
			map->find_by_basestation(pop_get_hostname(), keys, values);


			int walk = 6;
			int offset = -1;

			//		unsigned n = keys.size() ; // size before the inserts
			for( unsigned i = 0; i < keys.size(); i++ )
			{
				const PopChannelMap::PopChannelMapKey& key = keys[i];
				PopChannelMap::PopChannelMapValue val = values[i];
//				PopChannelMap::PopChannelMapValue updatedVal = val;


				if( val.tracker == zero_uuid || val.tracker == uuid ) // give slot to tracker if it's empty OR if we've already given it to the same tracker
				{
					if( offset == -1 )
					{
						offset = key.slot;
					}

					if( ((key.slot-offset) % walk) != 0 )
					{
						continue;
					}


					slots[chosen] = key.slot;
					chosen++;

					map->set(key.slot, uuid, val.basestation);


					if( chosen >= remaining )
					{
						break;
					}
				}
			}

			ota_packet_t packet;
			ota_packet_zero_fill(&packet);

			ostringstream os;
			os << "{\"result\":[";
			for( unsigned i = 0; i < chosen; i++ )
			{
				if( i != 0 )
				{
					os << ",";
				}
				os << slots[i];
			}
			os << "],\"id\":" << original_id << "}";

			snprintf(packet.data, sizeof(packet.data), "%s", os.str().c_str()); // lazy way to cap length
			ota_packet_prepare_tx(&packet);

			puts(packet.data);

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);

		}

//
//		for (std::vector<mystruct>::iterator iter = Vect.begin(); iter != Vect.end(); ++iter)
//		{
//			Vect.insert(iter + 1, otherstruct);
//
//		}

	}

	if( method.compare("poll") == 0 )
	{
		p0 = find_json_token(arr, "params[0]");
		p1 = find_json_token(arr, "params[1]");
		if( p0 && p0->type == JSON_TYPE_STRING && p1 && p1->type == JSON_TYPE_NUMBER )
		{
			std::string uuid_string = FROZEN_GET_STRING(p0);
			uuid_t uuid = b64_to_uuid(uuid_string);

			if( !basestation_should_respond(uuid) )
			{
				cout << "NOT REPLYING" << endl;
				return;
			}


			double pit_epoc = (double)pitPrnCodeStart/19200000.0;

			cout << "start: " << pitPrnCodeStart << endl;
//			printf("Epoc: %lf\r\n", pit_epoc);

			uint64_t tracker_pit = parseNumber<uint64_t>(FROZEN_GET_STRING(p1));

			cout << "BS PIT: " << pitPrnCodeStart << endl;
			cout << "T  PIT: " << tracker_pit << endl;
//			cout << "t slot: " <<

			// this trim includes jitter between the time that the tracker builds the packet with it's PIT value and the time when it's transmitted.
			// A better way is to determine which slot the tracker was trying to tx, calculate the pit counts which differ from that slot, and correct based on that
//			int64_t error_counts = tracker_pit - pitPrnCodeStart;

			uint32_t target_slot = pop_get_nearest_slot( uuid, pop_get_slot_pit_rounded(tracker_pit) );
			int64_t error_counts = pop_get_slot_error(target_slot, pitPrnCodeStart);



//			PopTimestamp system_now = get_microsec_system_time();
//			uint64_t system_pit = round(system_now.get_frac_secs() * 19200000.0) + system_now.get_full_secs()*19200000;

			cout << "target_slot: " << target_slot << endl;
			cout << "error_counts: " << error_counts << endl;
//			cout << "System Slot: " << pop_get_slot_pit_rounded(system_pit) << endl;
//			cout << "Closest slot: " << pop_get_tracker_slot_now(uuid) << endl;

			ota_packet_t packet;
			ota_packet_zero_fill(&packet);

			ota_packet_t* queued_packet;

			// check if there is anything queued up
			queued_packet = peek_packet(uuid_string);


			if( queued_packet )
			{
				memcpy(&packet, queued_packet, sizeof(ota_packet_t));
				erase_packet(uuid_string, *queued_packet);
			}
			else if( abs(error_counts) > 0.05 * 19200000.0 )
			{
				ostringstream os;
				os << "{\"method\":\"trim_utc\",\"params\":[" << -1*error_counts << "]}";

				snprintf(packet.data, sizeof(packet.data), "%s", os.str().c_str()); // lazy way to cap length

				cout << endl;
				puts(packet.data);
				cout << endl << endl;
			}
			else
			{
				// do nothing
				snprintf(packet.data, sizeof(packet.data), "{}");
			}






			ota_packet_prepare_tx(&packet);

			rpc->packet_tx((char*)(void*)&packet, packet.size, txTime, pitTxTime);
		}
	}
}

// most of this was copied from popjsonrpc.cpp
void PopPacketHandler::process_ota_packet(ota_packet_t* p, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart)
{
	const char *json = p->data;

	struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS];
	const struct json_token *methodTok = 0, *paramsTok = 0, *idTok = 0;

	// Tokenize json string, fill in tokens array
	int returnValue = parse_json(json, strlen(json), arr, POP_JSON_RPC_SUPPORTED_TOKENS);

	if( returnValue == JSON_STRING_INVALID || returnValue == JSON_STRING_INCOMPLETE )
	{
		cout << "problem with json string (" <<  json << ")" << endl;
		return;
	}

	if( returnValue == JSON_TOKEN_ARRAY_TOO_SMALL )
	{
		cout << "problem with json string (too many things for us to parse)" << endl;
		return;
	}

	// verify message has "method" key
	methodTok = find_json_token(arr, "method");
	if( !(methodTok && methodTok->type == JSON_TYPE_STRING) )
	{
		return;
	}

	// verify message has "params" key
	paramsTok = find_json_token(arr, "params");
	if( !(paramsTok && paramsTok->type == JSON_TYPE_ARRAY) )
	{
		return;
	}

	// "id" key is optional.  It's absence means the message will not get a response
	idTok = find_json_token(arr, "id");
	if( !(idTok && idTok->type == JSON_TYPE_NUMBER) )
	{
		idTok = 0;
	}

	execute(methodTok, paramsTok, idTok, arr, p->data, txTime, pitTxTime, pitPrnCodeStart);
}

uint32_t comb[] = {0, 58080, 92400, 100320, 126720, 134640, 155760, 158400, 166320, 168960, 171600, 190080, 198000, 205920, 208560, 229680, 234960, 248160, 253440, 274560, 282480, 300960, 314160, 319440, 322080, 327360, 348480, 353760, 361680, 364320, 366960, 374880, 390720, 403920, 406560, 411840, 425040, 477840, 512160, 546480, 567600, 612480, 623040, 633600, 638880, 657360, 665280, 689040, 707520, 715440, 723360, 733920, 736560, 765600, 778800, 789360, 813120, 815760, 852720, 871200, 876480, 881760, 918720, 939840, 971520, 1003200, 1037520, 1063920, 1074480, 1082400, 1100880, 1111440, 1153680, 1190640, 1195920, 1198560, 1203840, 1211760, 1227600, 1261920, 1264560, 1267200, 1290960, 1314720, 1317360, 1322640, 1351680, 1386000, 1412400, 1417680, 1430880, 1457280, 1481040, 1491600, 1496880, 1515360, 1525920, 1528560, 1539120, 1576080, 1594560, 1605120, 1626240, 1647360, 1673760, 1689600, 1710720};

void PopPacketHandler::process(const uint32_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	// from here below we are reading timer values
	boost::mutex::scoped_lock lock(timer_mtx);


	static uint32_t total_samples = 0;
	total_samples += size;

//	if( total_samples > 6000000 )
//	{
//		total_samples = 0;
//		std::string msg = "{\"method\":\"tmr_sync\",\"params\":[]}";
//		rpc->send_rpc(msg);
//	}

	if( total_samples < 1500*10 )
	{
		cout << "got " << size << " samples" << endl;
	}

	uint32_t combDenseLength = comb[ARRAY_LEN(comb)-1];


	size_t i,j;
	uint32_t prnCodeStart, bitSyncStart;

	int32_t scorePrn, scoreBitSync;

	boost::timer t; // start timing


	static uint32_t previous_run_offset = 0;

	// was was the # of the last sample we took
	uint32_t final_sample = 0;

	data -= previous_run_offset;
	size += previous_run_offset;




	if( new_timers == 0 )
	{
		return;
	} else if( artemis_tpm_start == -1 )
	{
//		for(i = 0; i < size;i++)
//		{
//			if( data[i] > artemis_tpm )
//			{
//				artemis_tpm_start = 0;
//				previous_run_offset = size-i;
//				return;
//			}
//		}
//
//		// Havent hit start condition yet
//		artemis_tpm_start = 0;
//		previous_run_offset = 0;
//		return;
	}





	for(i = 1; i<size;i++)
	{
		// the data wraps
		if( data[i-1] > data[i] && ((((uint64_t) data[i] + 0xffffffff) - ((uint64_t) data[i-1])) < (ARTEMIS_CLOCK_SPEED_HZ*0.1) ) )
		{
			// nothing can handle a wrap yet, so just bail and try again
			cout << "Data wrap edge condition" << endl;

			previous_run_offset = size-i;
			return;
		}
	}



	prnCodeStart = pop_correlate_spool(data, size, comb, ARRAY_LEN(comb), &scorePrn, &final_sample);

	previous_run_offset = size - final_sample;

	int temp = 0;

	if( temp )
	{
		for(j = 0; j<size; j++)
		{
			printf("%u, ", data[j]);
		}
	}



//	printf("Score: %d\r\n", scorePrn);
//	printf("Start: %d\r\n", prnCodeStart);
//	if( abs(scorePrn) < )

	double elapsed_time = t.elapsed();

//	if( prnCodeStart == 0 || elapsed_time > 4.0 )
//	{
//		printf("\r\n");
//
//		for(j = 0; j<size-1;j++)
//		{
//			printf("%u, ", data2[j]);
//		}
//
////		pop_correlate(data2, size-1, comb, ARRAY_LEN(comb), &scorePrn);
//
//	}

//	printf("time %f\r\n", elapsed_time);



	// keep the artemis_tpm, artemis_pit, artemis_pps counters no more than 4 seconds behind
//	for(i = 0; i < size; i++)
//	{
//		//uint32_t mod = data[i] - artemis_tpm;
//		if( ((uint32_t)(data[i] - artemis_pps)) > (ARTEMIS_CLOCK_SPEED_HZ*4) )
//		{
////			cout << "    bump from: " << artemis_tpm << " to " << (artemis_tpm + ARTEMIS_CLOCK_SPEED_HZ) << " to data[" << i << "]: " << data[i] << endl;
//			//cout << "  mod: " << mod << endl;
//
//			//cout << "bump with: " << data[i] << endl;
//
//			// bump all the counters
//			artemis_tpm += ARTEMIS_CLOCK_SPEED_HZ;
//			artemis_pit += ARTEMIS_PIT_SPEED_HZ;
//			artemis_pps += ARTEMIS_CLOCK_SPEED_HZ;
//			artimes_pps_full_sec++;
//		}
//
//	}

	//cout << "sec: " << artimes_pps_full_sec << endl;





	if( prnCodeStart != 0 )
	{
		printf("Score: %d\r\n", scorePrn);
		cout << "prnCodeStart: " << prnCodeStart << endl;

		short flag1 = 0, flag2 = 0;
		for(i = 1; i < size; i++)
		{
			if( data[i] > (prnCodeStart+combDenseLength) && !flag1 )
			{
				flag1 = 1;
			}
		}

		if( !flag1 )
		{
//			printf("\r\n");
//			for(i=0;i<(size);i++)
//			{
////				data[i] = (uint32_t)data[i];
//				printf("%u, ", data[i]);
//			}
//			printf("\r\n\r\n");
//
			printf("FIXME: matched comb in this chunk, but we need to wait till next process() to get everything...");
			printf("data was not longer than comb %d %d\r\n", data[size], (prnCodeStart+combDenseLength) );

			return;
		}

		uint32_t artemis_tpm;
		uint64_t artemis_pit;
		uint32_t artemis_pps;
		uint64_t artimes_pps_full_sec;

		int flag = 0;
		for (auto timer = artemis_timers.rbegin(); timer != artemis_timers.rend(); ++timer)
		{
			boost::tie(artemis_tpm, artemis_pit, artemis_pps, artimes_pps_full_sec) = *timer;
//			cout << "pps: " << prnCodeStart - artemis_pps << endl;
			if( (prnCodeStart - artemis_pps) < 48000000 )
			{
//				cout << "yes";
				flag = 1;
				break;
			}
		}

		if( flag == 0 )
		{
			cout << "couldn't find pps near enough!!" << endl;
		}



		cout << endl;

		for(i = 0; i < size; i++)
		{
			cout << data[i] << ",";
		}

		cout << endl;




		uint32_t artemis_tpm2 = artemis_tpm;
		uint64_t artemis_pit2 = artemis_pit;
		uint32_t artemis_pps2 = artemis_pps;
		uint64_t artimes_pps_full_sec2 = artimes_pps_full_sec;

//		// now that we have an actual start of frame, update these as aggressively as possible
//		while( ((uint32_t)(prnCodeStart - artemis_pps2)) > ARTEMIS_CLOCK_SPEED_HZ )
//		{
//			cout << "JIT bump from: " << artemis_tpm2 << " to " << (artemis_tpm2 + ARTEMIS_CLOCK_SPEED_HZ) << endl;
//
//			// bump all the counters
//			artemis_tpm2 += ARTEMIS_CLOCK_SPEED_HZ;
//			artemis_pit2 += ARTEMIS_PIT_SPEED_HZ;
//			artemis_pps2 += ARTEMIS_CLOCK_SPEED_HZ;
//			artimes_pps_full_sec2++;
//		}



//		uint32_t lastTime = data[size-1];
		//
		// pit_last_sample_time this is approximately the time of the last sample from the dma

		// this is approximately the pit time of start of frame
//		uint64_t pitPrnCodeStart = pitLastSampleTime - ((lastTime - prnCodeStart)*(double)ARTEMIS_PIT_SPEED_HZ/(double)ARTEMIS_CLOCK_SPEED_HZ);


		uint32_t delta_counts = prnCodeStart - artemis_tpm2;

		if( delta_counts > ARTEMIS_CLOCK_SPEED_HZ )
		{
			cout << "    delta_counts: " << delta_counts << endl;
		}


//		 artemis_tpm << " pit: " << artemis_pit << " pps: " << artemis_pps
		// timers are syncronized which gives maching values at the same time
		// (tpm start of frame - last synced tpm) over tpm period times pit period = delta pit counts
		uint64_t pitPrnCodeStart = ( (delta_counts) / 48000000.0 ) * 19200000.0;

		// offset to get absolute counts
		pitPrnCodeStart += artemis_pit2;

		cout << "pitPrnCodeStart: " << pitPrnCodeStart << endl;

//		double pit_epoc = (double)pitPrnCodeStart/19200000.0;
//		static double pit_epoc_last;

//		printf("PIT start: %lf\r\n", pit_epoc);
//		printf("PIT delta: %lf\r\n", pit_epoc-pit_epoc_last);


//		pit_epoc_last = pit_epoc;

		double txDelta = 0.75;



		// add .75 seconds
		uint32_t txTime = (prnCodeStart + (uint32_t)(ARTEMIS_CLOCK_SPEED_HZ*txDelta)) % ARTEMIS_CLOCK_SPEED_HZ;

		uint64_t pitTxTime = pitPrnCodeStart + (uint32_t)(ARTEMIS_PIT_SPEED_HZ*txDelta);







		//cout << "pit delta: " << ( pitTxTime - pitPrnCodeStart ) << endl;











		ota_packet_t rx_packet;

		unsigned peek_length = 4; // how many bytes do we need to find the size?
		unsigned peek_length_encoded = ota_length_encoded(peek_length);

		uint8_t data_rx[peek_length_encoded];


		shannon_pop_data_demodulate(data, size, prnCodeStart+combDenseLength, data_rx, peek_length_encoded, (scorePrn<0?1:0));

		uint8_t data_decode[peek_length];

		uint32_t data_decode_size;

		decode_ota_bytes(data_rx, peek_length_encoded, data_decode, &data_decode_size);


		//	printf("data: %02x\r\n", data_decode[0]);
		//	printf("data: %02x\r\n", data_decode[1]);
		//	printf("data: %02x\r\n", data_decode[2]);
		//	printf("data: %02x\r\n", data_decode[3]);
		//

		ota_packet_zero_fill(&rx_packet);
		memcpy(&rx_packet, data_decode, peek_length);
		uint16_t packet_size = MIN(rx_packet.size, ARRAY_LEN(rx_packet.data)-1);

		int packet_good = 0;
		int j;

		// search around a bit till the checksum matches up.  This is our "bit sync"
		for(j = -5000; j < 5000; j+=50)
		{
			uint16_t decode_remainig_size = MAX(0, packet_size);
			unsigned remaining_length = ota_length_encoded(decode_remainig_size);
			uint8_t data_rx_remaining[remaining_length];
			shannon_pop_data_demodulate(data, size, prnCodeStart+combDenseLength+j, data_rx_remaining, remaining_length, (scorePrn<0?1:0));
			uint8_t data_decode_remaining[decode_remainig_size];
			uint32_t data_decode_size_remaining;
			decode_ota_bytes(data_rx_remaining, remaining_length, data_decode_remaining, &data_decode_size_remaining);

			// the null terminated character is not transmitted
			ota_packet_zero_fill_data(&rx_packet);
			memcpy(((uint8_t*)&rx_packet), data_decode_remaining, decode_remainig_size);

			if(ota_packet_checksum_good(&rx_packet))
			{
				packet_good = 1;
				break;
			}
		}

		if( !packet_good )
		{
			printf("Bad packet checksum\r\n");


//			printf("Packet (still) says: ");
//
//			for(int k = 0; k < 40; k++ )
//			{
//				char c = rx_packet.data[k];
//				if( isprint(c) )
//				{
//					cout << c;
//				}
//				else
//				{
//					cout << '0';
//				}
//			}
//
//			cout << endl;

//			return;
		}
		else
		{

			if(rx_packet.data[ARRAY_LEN(rx_packet.data)-1] != '\0' )
			{
				printf("Packet c-string is not null terminated\r\n");
				return;
			}

			printf("Packet (offset %d) says: %s\r\n", j, rx_packet.data);

			//		printf("Packet (offset %d) says: ", j);
			//
			//		for(int k = 0; k < 40; k++ )
			//		{
			//			char c = rx_packet.data[k];
			//			if( isprint(c) )
			//			{
			//				cout << c;
			//			}
			//			else
			//			{
			//				cout << '0';
			//			}
			//		}
			//
			//		cout << endl;



			cout << "tpm: " << artemis_tpm2 << " pit: " << artemis_pit2 << " pps: " << artemis_pps2 << endl;

			if( rpc )
			{
				process_ota_packet(&rx_packet, txTime, pitTxTime, pitPrnCodeStart);
			}
			else
			{
				printf("Rpc pointer not set, skipping json parse\r\n");
			}
			printf("\r\n");

		} // packet good


		// regardless if checksum was good or bad, we've got a comb here.  now time to do fabric stuff (which is blocking?) after transmitting reply which is time sensative


		uint32_t rx_frac_int = (prnCodeStart - artemis_pps2);

		ostringstream os;

		os << "{\"method\":\"packet_rx\",\"params\":[" << "\"" << pop_get_hostname() << "\"" << "," << artimes_pps_full_sec2 << "," << rx_frac_int << "]}";

		rpc->fabric->send("noc", os.str());



	} // comb detected


}

void PopPacketHandler::set_artimes_timers(uint32_t a_tpm, uint64_t a_pit, uint32_t a_pps)
{
	 boost::mutex::scoped_lock lock(timer_mtx);

	 uint32_t artemis_tpm;
	 uint64_t artemis_pit;
	 uint32_t artemis_pps;
	 uint64_t artimes_pps_full_sec;

	 artemis_tpm = a_tpm;
	 artemis_pit = a_pit;
	 artemis_pps = a_pps;

	 new_timers++;
	 artemis_tpm_start = -1;



	 // "now" in system time
	 PopTimestamp now = get_microsec_system_time();

	 // artemis_tpm is the most recent timer value, sent to us over uart asap.  we assume this is pretty accurately "now"

	 // this is how far we were into the second when we took a time reading.
	 double frac_since_pps = ((double)(artemis_tpm - artemis_pps))/ARTEMIS_CLOCK_SPEED_HZ;

	 // make timestamp
	 PopTimestamp delta_since_pps(frac_since_pps);

	 // offset as if we had taken system time reading at the edge of pps
	 now -= delta_since_pps;

	 // round
	 artimes_pps_full_sec = round(now.get_real_secs());


	 artemis_timers.push_back(boost::make_tuple(artemis_tpm, artemis_pit, artemis_pps, artimes_pps_full_sec));



//	 cout << "time delta: " << artemis_tpm - artemis_pps << endl;

}


} //namespace

