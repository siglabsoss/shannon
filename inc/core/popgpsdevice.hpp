#ifndef __POP_GPS_DEVICE_HPP_
#define __POP_GPS_DEVICE_HPP_


#include <boost/tuple/tuple.hpp>
#include "core/popsource.hpp"
#include "core/popparsegps.hpp"
#include "core/objectstash.hpp"


using namespace std;

namespace pop
{


class PopGpsDevice: public PopSink<boost::tuple<char[20], PopTimestamp>>
{
public:
	PopSource<char> tx;
	PopParseGPS *gps;
	ObjectStash radios;

	PopGpsDevice(size_t chunk);

	unsigned int run_loop();
	void init();
	void mock();
	void process(const boost::tuple<char[20], PopTimestamp>* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);

};

}


#endif
