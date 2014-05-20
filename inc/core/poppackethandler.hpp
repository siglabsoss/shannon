#ifndef __POP_PACKET_HANDLER_HPP_
#define __POP_PACKET_HANDLER_HPP_

#include <string>
#include <boost/tuple/tuple.hpp>
#include <stdint.h>

#include "core/popsink.hpp"


namespace pop
{

class PopArtemisRPC;

class PopPacketHandler : public PopSink<uint64_t>
{
public:
       PopPacketHandler(unsigned notused);
       void process(const uint64_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
       void init() {}
       PopArtemisRPC* rpc;

};

}


#endif
