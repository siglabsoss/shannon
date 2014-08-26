#ifndef __POP_SERIAL_HPP_
#define __POP_SERIAL_HPP_

#include <string>     // string function definitions
#include <iostream>
#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "boost_serial/AsyncSerial.h"

using namespace std;



namespace pop
{

class PopSerial : public PopSink<char>
{
public:
	PopSink<char> *tx;
	PopSource<char> rx; // serial receive generates characters
	string path;
	bool print_all;

	// http://www.webalice.it/fede.tft/serial_port/serial_port.html
	// https://gitorious.org/serial-port/serial-port/source/03e161e0b788d593773b33006e01333946aa7e13:
	CallbackAsyncSerial handle;

	PopSerial(std::string devicePath, unsigned baud, const char* name, bool print_all = 0);
	~PopSerial();
	void characters_received(const char*, size_t);
	void init();
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
};

}


#endif
