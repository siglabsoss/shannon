#include "core/popfabricbridge.hpp"

#include <stddef.h>

#include <iostream>
#include <string>


using namespace std;



namespace pop
{



PopFabricBridge::PopFabricBridge(PopFabric *f, std::string d) : PopSink<char>("PopFabricBridgeRx", 1), tx("PopFabricBridgeTx"), headValid(0), fabric(f), destination(d)
{
	if( fabric )
	{
		fabric->set_receive_function(boost::bind(&PopFabricBridge::fabric_rx, this, _1, _2, _3));
	}
}

void PopFabricBridge::fabric_rx(std::string to, std::string from, std::string msg)
{
	cout << "to: " << to << " from: " << from << " msg: " << msg << endl;

	unsigned length = msg.length() + 2;

	char bytes[length];

	bytes[0] = 0;

	strncpy(bytes+1, msg.c_str(), msg.length() );

	bytes[length-1] = 0;

	this->tx.process(bytes, length);
}

void PopFabricBridge::init() {}

void PopFabricBridge::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	if( size != 1 ) {
		cout << "Error " << this->get_name() << " may only accept 1 character at a time";
		return;
	}

	char c = data[0];

//	cout << c;

	// store characters until a null arrives
	// forward to fabric once conditions are valid
	if( !headValid )
	{
		if( c == 0 )
			headValid = true;
	}
	else
	{
		if( c == 0 )
		{
			std::string msg(command.begin(),command.end());

			fabric->send(destination, msg);

			command.erase(command.begin(),command.end());
		}
		else
		{
			command.push_back(c);
		}
	}
}

}
