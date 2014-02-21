#include "core/popgpsdevice.hpp"
#include <iostream>
#include <boost/asio.hpp>

using namespace std;



namespace pop
{

PopGpsDevice::PopGpsDevice(size_t chunk) : tx("PopSerialSource"), gps(0)
{
	this->tx.set_loop_function(boost::bind(&PopGpsDevice::run_loop,this));

	double lat = 37;
	double lon = -122;

	PopRadio* r = radios[10000]; // find or create
	r->setLat(lat);
	r->setLng(lon);
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
		cout << "got gpx fix " << lat << ", " << lng << endl;
	}

//	unsigned char buf[3] = "{}";
//	tx.process(buf, 2);
	std::string json = this->radios[10000]->seralize();
//	cout <<  << endl;
//	json.length();

	char nul = 0;

	// this should be wrapped into a single call to send a single packet
	tx.process(&nul, 1);
	tx.process(json.c_str(), json.length());
	tx.process(&nul, 1);


	boost::posix_time::milliseconds workTime(1000);
	boost::this_thread::sleep(workTime);

	return 0;
}


} //namespace

