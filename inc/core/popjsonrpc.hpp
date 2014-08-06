#ifndef __POP_JSON_RPC_HPP_
#define __POP_JSON_RPC_HPP_


#include <stddef.h>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "frozen/frozen.h"
#include "core/utilities.hpp"

// how many tokens do we support in messages
#define POP_JSON_RPC_SUPPORTED_TOKENS (2000)


namespace pop
{


// This class handles tx/rx of PopWi's JSON RPC implementation.  The PopSink that this class extends is the "rx"
class PopJsonRPC : public PopSink<char>
{
public:
	// This is the "tx" direction and responds to messages with JSON character responses
	PopSource<char> tx;
	bool headValid;
	bool rawMode;
	std::vector<char> command;


	PopJsonRPC(unsigned notused);
	void init();
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void parse();
	void rcp_log(std::string log);

    /**
     * Needs to be implemented by child class to handle the mapping between strings and function calls
     */
	virtual void execute_rpc(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str) = 0;
	virtual void execute_result(const struct json_token *resultTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str) = 0;
	virtual void execute_raw(char c) = 0;

	void respond_int(int value, int methodId);
	void send_rpc(const char *rpc_string, size_t length);
	void send_rpc(std::string& rpc);
	uint16_t rpc_get_autoinc(void);
};

}

#endif
