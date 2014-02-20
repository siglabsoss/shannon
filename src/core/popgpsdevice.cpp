#include "core/popgpsdevice.hpp"
#include <iostream>
#include <boost/asio.hpp>

using namespace std;



namespace pop
{

PopGpsDevice::PopGpsDevice(size_t chunk) : tx("PopSerialSource"), gps(0)
{
	this->tx.set_loop_function(boost::bind(&PopGpsDevice::run_loop,this));
}


// this gets called over and over in a while(1), this function should not contain any sort of infinite loop itself
// return non-zero to exit loop
unsigned int PopGpsDevice::run_loop()
{
	double lat,lng,time;
	static int i = 0;

	cout << i++ << endl;

	if( this->gps->gpsFixed() )
	{

		boost::tie(lat, lng, time) = this->gps->getFix();
		cout << "got gpx fix " << lat << ", " << lng << << endl;
	}


	boost::posix_time::milliseconds workTime(1000);
	boost::this_thread::sleep(workTime);

	return 0;
}



void PopGpsDevice::process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{

}


} //namespace

