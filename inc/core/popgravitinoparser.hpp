


#ifndef __POP_GRAV_PARSER__
#define __POP_GRAV_PARSER__

#include <stdint.h>

#include <boost/tuple/tuple.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "core/poppackethandler.hpp"
#include "core/popfabric.hpp"


namespace pop
{

class PopSightingStore;

// This class handles all RPC between the gravitino basestations and the s3p
class PopGravitinoParser
{
public:
	std::vector<char> command;

	std::vector<std::vector<char> > streams;
	std::vector<std::string> remotes;

	PopGravitinoParser(PopSightingStore* sighting_store, PopFabric* f);

	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str, unsigned stream_index);
	void parse(unsigned);
	void rcp_log(std::string log);
	uint16_t rpc_get_autoinc(void);
	unsigned get_stream_index(std::string &name);
	void send_fabric_rpc(const char* data, size_t size, unsigned stream_index);
	void fabric_rx(std::string, std::string, std::string);

private:
	PopFabric* fabric;
	PopSightingStore* const sighting_store_;
};

}

#endif
