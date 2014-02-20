#ifndef __POP_GPS_DEVICE_HPP_
#define __POP_GPS_DEVICE_HPP_


#include <core/popsource.hpp>

#include "core/popparsegps.hpp"


using namespace std;

namespace pop
{


class PopGpsDevice
{
public:
	PopSource<unsigned char> tx;
	PopParseGPS *gps;

	PopGpsDevice(size_t chunk);
	void init()
	{
		//    	 handle.writeString("Serial Boot\r\n");
		//    	while(1)
		//    	{
		//    		cout<<"Received : " << handle.readChar() << " : end" << endl;
		//    	}

		//		handle.close();
	}

	unsigned int run_loop();
	void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);

};

}


#endif
