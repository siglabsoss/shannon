#ifndef __POP_PACKET_HANDLER_HPP_
#define __POP_PACKET_HANDLER_HPP_

#include <core/popsink.hpp>
#include <string>
#include <boost/tuple/tuple.hpp>
#include <stdint.h>


namespace pop
{


class PopPacketHandler : public PopSink<uint32_t>
{
public:
//     bool headValid;
//     std::vector<unsigned char> command;
//     bool gpsFix;
//     double lat;
//     double lng;
//     boost::mutex mtx_;

public:
       PopPacketHandler(unsigned notused);
       void process(const uint32_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
       void init() {}
//     void gga(std::string &str);
//     void parse();
//     bool gpsFixed();
//     boost::tuple<double, double, double> getFix();

private:
//     void setFix(double lat, double lng, double time);


};

}


#endif
