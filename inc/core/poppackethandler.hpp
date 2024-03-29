#ifndef __POP_PACKET_HANDLER_HPP_
#define __POP_PACKET_HANDLER_HPP_

#include <string>
#include <boost/tuple/tuple.hpp>
#include <stdint.h>

#include "core/popsink.hpp"
#include "dsp/prota/popsparsecorrelate.h"
#include "popjsonrpc.hpp"
#include "core/pops3prpc.hpp"
#include "core/popchannelmap.hpp"


namespace pop
{

class PopArtemisRPC;
class PopS3pRPC;
class PopChannelMap;

class PopPacketHandler : public PopSink<uint64_t>
{
public:
       PopPacketHandler(unsigned notused);
       void process(const uint64_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
       void process_ota_packet(ota_packet_t* p, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart);
       void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], char *str, uint32_t txTime, uint64_t pitTxTime, uint64_t pitPrnCodeStart);
       int32_t pop_get_tracker_slot_now(uuid_t uuid);
       int32_t pop_get_nearest_slot(uuid_t uuid, int32_t system_now_slot);
       int basestation_should_respond(uuid_t uuid);
       void init() {}
       PopArtemisRPC* rpc;
       PopS3pRPC* s3p;
       PopChannelMap* map;

};

}


#endif
