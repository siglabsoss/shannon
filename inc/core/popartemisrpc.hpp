#ifndef __POP_ARTEMIS_RPC_HPP_
#define __POP_ARTEMIS_RPC_HPP_

#include <stdint.h>
#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/thread/mutex.hpp>

#include "core/popjsonrpc.hpp"
#include "core/poppackethandler.hpp"
#include "core/popfabric.hpp"


namespace pop
{


// This class handles all RPC between the Artemis board and Gravitino instance of shannon on the BBB.  Character tx/rx are handled by the parent class PopJsonRPC.
// The member function execute() is called by the parent class when valid json RPC is received.  It then generates binary objects on the "packet" PopSource
class PopArtemisRPC : public PopJsonRPC
{
public:
	PopArtemisRPC(PopFabric*, std::string attached = "");
	PopSource<boost::tuple<char[20], PopTimestamp>> packets;
	PopSource<uint32_t> edges;

	void execute_rpc(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);
	void execute_result(const struct json_token *resultTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);
	void execute_csv(std::string str);
	void packet_tx(char* data, uint32_t size, uint32_t txTime, uint64_t pitTxTime);
	void send_reset();
	void set_role_base_station();
	void mock(void);
	void mock_csv(void);
	int received_basestation_boot();
	void fabric_rx(std::string to, std::string from, std::string msg);

	PopPacketHandler* handler;
	int basestation_boot;
	std::string attached_uuid;
	PopFabric *fabric;

	std::vector<int32_t> rpc_ids;
	mutable boost::mutex rpc_ids_mtx;
};

}

#endif
