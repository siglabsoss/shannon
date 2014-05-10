#ifndef __POP_PARSE_GPS_HPP_
#define __POP_PARSE_GPS_HPP_

#include <core/popsink.hpp>
#include <string>
#include <boost/tuple/tuple.hpp>

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
	boost::mutex mtx_;

public:
	PopParseGPS(unsigned notused);
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void init() {}
	void gga(std::string &str);
	void parse();
	bool gpsFixed();
	boost::tuple<double, double, double> getFix();

private:
	void setFix(double lat, double lng, double time);


};

}


#endif
