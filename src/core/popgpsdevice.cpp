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

}




void PopGpsDevice::init() {}

void PopGpsDevice::mock()
{
	// basestation name, lat, lon, device serial, full seconds, frac seconds
	std:string msg = "{\"method\":\"bx_rx\",\"params\":[\"dss-Aspire-4830TG\",37.476441,-122.180114,1,1400554351,0.00202]}";

	std::stringstream buildPacket;
	buildPacket << '\0' << msg << '\0';
	std::string packet = buildPacket.str();

	tx.process(packet.c_str(), packet.length());
}

void PopGpsDevice::process(const boost::tuple<char[20], PopTimestamp>* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{


}


} //namespace

