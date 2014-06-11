#include <iostream>
#include <boost/asio.hpp>
#include <unistd.h>
#include "json/json.h"
#include "core/popgpsdevice.hpp"
#include "core/config.hpp"

using namespace std;



namespace pop
{

PopGpsDevice::PopGpsDevice(size_t chunk) : PopSink<boost::tuple<char[20], PopTimestamp>>("PopDevicePacketSink", 1), tx("PopSerialSource"), gps(0)
{
//	this->tx.set_loop_function(boost::bind(&PopGpsDevice::run_loop,this));
//
//	double lat = 37;
//	double lon = -122;
//
//	PopRadio* r = radios[10000]; // find or create
//	r->setLat(lat);
//	r->setLng(lon);
}


// this gets called over and over in a while(1), this function should not contain any sort of infinite loop itself
// return non-zero to exit loop
//unsigned int PopGpsDevice::run_loop()
//{
//	// sleep first because we optionally return early in this function
//	boost::posix_time::milliseconds workTime(1000);
//	boost::this_thread::sleep(workTime);
//
//	double lat,lng,time;
//	static int i = 0;
//
//	//cout << i++ << endl;
//
//	if( !this->gps->gpsFixed() )
//	{
//		return 0; // bail with 0 which means we still want to loop
//	}
//
//	boost::tie(lat, lng, time) = this->gps->getFix();
//	cout << "got gpx fix " << lat << ", " << lng << endl;
//
//	PopRadio *r = this->radios[10000];
//
//	r->setLat(lat);
//	r->setLng(lng);
//
//	std::string json = r->seralize();
////	cout << json.c_str() << endl;
////	json.length();
//
//	char nul = 0;
//
//	std::stringstream buildPacket;
//	buildPacket << nul << json << nul;
//	std::string packet = buildPacket.str();
//
//	tx.process(packet.c_str(), packet.length());
//
//	return 0;
//}


void PopGpsDevice::init() {}

void PopGpsDevice::process(const boost::tuple<char[20], PopTimestamp>* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{

	bool allow_unfixed = false;

	try
	{
		allow_unfixed = rbx::Config::get<bool>("allow_gps_unfixed");
	}
	catch(std::exception const& x) {}

	if( !this->gps->gpsFixed() && !allow_unfixed )
	{
		return; // bail
	}


	double lat,lng,time;
	static int i = 0;


	boost::tie(lat, lng, time) = this->gps->getFix();
	cout << "got gpx fix " << lat << ", " << lng << endl;



	char hostname[256];
	int ret = gethostname(hostname, 256);
	if( ret != 0 )
	{
		cout << "couldn't read linux hostname!" << endl;
		strncpy(hostname, "unkown", 256);
	}

	// construct programatically
	json::array params;
	params.append(hostname);
	params.append(lat);
	params.append(lng);
	params.append(get<0>(data[0]));
	params.append(get<1>(data[0]).get_full_secs());
	params.append(get<1>(data[0]).get_frac_secs());


	json::object o;
	o.insert("method", "bs_rx");
	o.insert("params", params);

//	cout << json::serialize(o) << endl;

	std::stringstream buildPacket;
	buildPacket << '\0' << json::serialize(o) << '\0';
	std::string packet = buildPacket.str();

	tx.process(packet.c_str(), packet.length());
}

void PopGpsDevice::greet_s3p(void)
{
	std::string message("{method:\"gravitino_boot\",\"params\":[]}");

	tx.process(message.c_str(), message.length());
}

} //namespace

