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
	// sleep first because we optionally return early in this function
	boost::posix_time::milliseconds workTime(1000);
	boost::this_thread::sleep(workTime);

	double lat,lng,time;
	static int i = 0;

	//cout << i++ << endl;

	if( !this->gps->gpsFixed() )
	{
		return 0; // bail with 0 which means we still want to loop
	}

	boost::tie(lat, lng, time) = this->gps->getFix();
	cout << "got gpx fix " << lat << ", " << lng << endl;

	PopRadio *r = this->radios[10000];

	r->setLat(lat);
	r->setLng(lng);

	std::string json = r->seralize();
//	cout << json.c_str() << endl;
//	json.length();

	char nul = 0;

	std::stringstream buildPacket;
	buildPacket << nul << json << nul;
	std::string packet = buildPacket.str();

	tx.process(packet.c_str(), packet.length());



	return 0;
}


} //namespace

