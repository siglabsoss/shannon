#include "popradio.h"
#include "json/json.h"

namespace pop
{

PopRadio::PopRadio()
{

}

PopRadio::~PopRadio()
{

}


std::string PopRadio::seralize()
{

	// construct programatically
	json::object o;
	o.insert("lat", lat)
            		 .insert("lng", lng)
            		 .insert("bat_current", bat_current)
            		 .insert("bat_voltage", bat_voltage)
            		 .insert("serial", serial)
            		 .insert("temp", temp)
            		 .insert("status", status);

	//
//	std::cout << json::pretty_print(o) << std::endl;

	return json::serialize(o);
}



float PopRadio::getBatCurrent() const {
	return bat_current;
}

void PopRadio::setBatCurrent(float batCurrent) {
	bat_current = batCurrent;
}

float PopRadio::getBatVoltage() const {
	return bat_voltage;
}

void PopRadio::setBatVoltage(float batVoltage) {
	bat_voltage = batVoltage;
}

double PopRadio::getLat() const {
	return lat;
}

void PopRadio::setLat(double lat) {
	this->lat = lat;
}

double PopRadio::getLng() const {
	return lng;
}

void PopRadio::setLng(double lng) {
	this->lng = lng;
}

long PopRadio::getSerial() const {
	return serial;
}

void PopRadio::setSerial(long serial) {
	this->serial = serial;
}

int PopRadio::getStatus() const {
	return status;
}

void PopRadio::setStatus(int status) {
	this->status = status;
}

float PopRadio::getTemp() const {
	return temp;
}

void PopRadio::setTemp(float temp) {
	this->temp = temp;
}

}
