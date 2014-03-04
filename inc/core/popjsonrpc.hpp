#ifndef __POP_JSON_RPC_HPP_
#define __POP_JSON_RPC_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>
#include "frozen/frozen.h"




namespace pop
{


class PopJsonRPC : public PopSink<unsigned char>
{
public:
	 PopSink<unsigned char> *tx; // PopSink must be inherited b/c of virtual classes, so we fake out a pointer here
	 PopSource<unsigned char> rx;
	 bool headValid;
	 std::vector<unsigned char> command;

//private:
	// pointer to a single timestamp given to us in the previous call to process
//	const PopTimestamp* previous_timestamp;
//
//	// the number of data samples from the previous call to process
//	size_t previous_size;
//public:
	PopJsonRPC(unsigned notused);
    void init();
    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);

    void parse();


    void execute(std::string &method, json_token *tokens, int methodId);


    void respond_int(int value, int methodId);
};

}


#endif
