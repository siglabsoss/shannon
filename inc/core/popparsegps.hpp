#ifndef __POP_PARSE_GPS_HPP_
#define __POP_PARSE_GPS_HPP_

#include <string>
#include <boost/tuple/tuple.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"

namespace pop
{


class PopParseGPS : public PopSink<char>
{
public:
	bool headValid;
	std::vector<char> command;
	bool gpsFix;
	double lat;
	double lng;
	PopSource<char> tx;
	boost::mutex mtx_;

public:
	PopParseGPS(unsigned notused);
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void init() {}
	void gga(std::string &str);
	void parse();
	bool gpsFixed();
	boost::tuple<double, double, double> getFix();
	void hot_start();
	boost::tuple<uint32_t, uint32_t> gps_now();
	std::string get_checksum(std::string str);
	void set_debug_on();
	void set_debug_off();

private:
	void setFix(double lat, double lng, double time);


};

}


#endif
