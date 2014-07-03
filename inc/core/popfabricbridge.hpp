#ifndef __POP_FABRIC_BRIDGE_HPP__
#define __POP_FABRIC_BRIDGE_HPP__


#include <stddef.h>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "core/popfabric.hpp"
#include "core/utilities.hpp"


namespace pop
{

// This class bridges between a PopSink<char>/PopSource<char> pair and a ZMQ fabric object.  All outgoing messages are to a fixed destination, passed in the constructor
// Incoming characters are held until a null character is received, and then the buffer is flushed as a single ZMQ packet.  Padding nulls are added to incoming ZMQ packets, and then flushed out the PopSource
class PopFabricBridge : public PopSink<char>
{
public:
	// This is the "tx" direction and responds to messages with JSON character responses
	PopSource<char> tx;
	bool headValid;

	std::vector<char> command;


	PopFabricBridge(PopFabric *f, std::string destination);
	void init();
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void fabric_rx(std::string to, std::string from, std::string msg);

	PopFabric* fabric;

	std::string destination;
};

}

#endif
