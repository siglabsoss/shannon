#include <iostream>
#include <boost/timer.hpp>

#include "core/poppackethandler.hpp"
#include "core/util.h"
#include "dsp/prota/popsparsecorrelate.h"
#include "core/popartemisrpc.hpp"
#include "core/basestationfreq.h"


using namespace std;

namespace pop
{


PopPacketHandler::PopPacketHandler(unsigned notused) : PopSink<uint32_t>("PopPacketHandler", 1), rpc(0)
{

}

void PopPacketHandler::process(const uint32_t* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{

}


} //namespace

