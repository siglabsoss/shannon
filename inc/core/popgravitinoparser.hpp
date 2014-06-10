


#ifndef __POP_GRAV_PARSER__
#define __POP_GRAV_PARSER__

#include <stdint.h>

#include <boost/tuple/tuple.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "core/poppackethandler.hpp"
#include "net/popnetworkwrapped.hpp"


namespace pop
{

class PopSightingStore;

// This class handles all RPC between the gravitino basestations and the s3p
class PopGravitinoParser : public PopSink<char>
{
public:
	// This is the "tx" direction and responds to messages with JSON character responses
	PopSource<char> tx;
	bool headValid;
	std::vector<char> command;

	std::vector<std::vector<char> > streams;
	std::vector<wrapped_net_header_t> remotes;

	PopGravitinoParser(unsigned notused, PopSightingStore* sighting_store);
	//PopSource<boost::tuple<char[20], PopTimestamp>> packets;

	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str, unsigned stream_index);
//	void packet_rx(std::string b64_serial, uint32_t offset, double clock_correction);
//	void packet_tx(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime);
//	void set_role_base_station();
//	void mock(void);
	void init();
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void parse(unsigned);
	void rcp_log(std::string log);
	uint16_t rpc_get_autoinc(void);
	unsigned get_stream_index(wrapped_net_header_t &header);
	void send_network_rpc(const char* data, size_t size, unsigned stream_index);

	//PopPacketHandler* handler;

private:
	PopSightingStore* const sighting_store_;
};

}

#endif
