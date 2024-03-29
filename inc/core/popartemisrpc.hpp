#ifndef __POP_ARTEMIS_RPC_HPP_
#define __POP_ARTEMIS_RPC_HPP_

#include <stdint.h>

#include <boost/tuple/tuple.hpp>

#include "core/popjsonrpc.hpp"
#include "core/poppackethandler.hpp"


namespace pop
{


// This class handles all RPC between the Artemis board and Gravitino instance of shannon on the BBB.  Character tx/rx are handled by the parent class PopJsonRPC.
// The member function execute() is called by the parent class when valid json RPC is received.  It then generates binary objects on the "packet" PopSource
class PopArtemisRPC : public PopJsonRPC
{
public:
	PopArtemisRPC(unsigned notused);
	PopSource<boost::tuple<char[20], PopTimestamp>> packets;

	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);
	void packet_rx(std::string b64_serial, uint32_t offset, double clock_correction);
	void packet_tx(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime);
	void send_reset();
	void set_role_base_station();
	void mock(void);
	int received_basestation_boot();

	PopPacketHandler* handler;
	int basestation_boot;
};

}

#endif
