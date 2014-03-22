#ifndef __POP_GRAV_RPC_HPP_
#define __POP_GRAV_RPC_HPP_

#include "core/popjsonrpc.hpp"
#include <boost/tuple/tuple.hpp>


namespace pop
{


// This class handles all RPC between the Artemis board and gravitino instance on BBB.  character tx/rx are handled by the parent class PopJsonRPC.
// The member function execute() is called by the parent class when valid json RPC is received.  It then generates binary objects on the "packet" PopSource
class PopGravitinoRPC : public PopJsonRPC
{
public:
	PopGravitinoRPC(unsigned notused);
	PopSource<boost::tuple<char[20], PopTimestamp>> packets;

	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);
	void packet_rx(std::string b64_serial, uint32_t offset, double clock_correction);
};

}

#endif
