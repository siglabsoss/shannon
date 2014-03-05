#ifndef __POP_JSON_RPC_HPP_
#define __POP_JSON_RPC_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>
#include "frozen/frozen.h"

#define POP_JSON_RPC_SUPPORTED_TOKENS (50)


namespace pop
{


class PopJsonRPC : public PopSink<unsigned char>
{
public:
	PopSink<unsigned char> *tx; // PopSink must be inherited b/c of virtual classes, so we fake out a pointer here
	PopSource<unsigned char> rx;
	bool headValid;
	std::vector<unsigned char> command;


	PopJsonRPC(unsigned notused);
	void init();
	void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);

	void parse();


	void execute(const struct json_token *methodTok, const struct json_token *paramsTok, const struct json_token *idTok, struct json_token arr[POP_JSON_RPC_SUPPORTED_TOKENS], std::string str);


	void respond_int(int value, int methodId);
};

}


#endif
