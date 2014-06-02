


#ifndef __POP_GRAV_PARSER__
#define __POP_GRAV_PARSER__

#include <stdint.h>

#include <boost/tuple/tuple.hpp>

#include "core/popjsonrpc.hpp"
#include "core/poppackethandler.hpp"


namespace pop
{

class PopSightingStore;

// This class handles all RPC between the gravitino basestations and the s3p
class PopGravitinoParser : public PopJsonRPC
{
public:
	PopGravitinoParser(unsigned notused, PopSightingStore* sighting_store);
	//PopSource<boost::tuple<char[20], PopTimestamp>> packets;

	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);
	void packet_rx(std::string b64_serial, uint32_t offset, double clock_correction);
	void packet_tx(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime);
	void set_role_base_station();
	void mock(void);

	//PopPacketHandler* handler;

private:
	PopSightingStore* const sighting_store_;
};

}

#endif
