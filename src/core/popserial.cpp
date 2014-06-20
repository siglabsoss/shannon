#include "core/popserial.hpp"

using namespace std;

namespace pop
{


PopSerial::PopSerial(std::string devicePath, unsigned baud) : PopSink<char>("PopSerialSink", 0), rx("PopSerialSource"), path(devicePath), handle(devicePath, baud)
{
	// note the weird _1 and _2. see http://stackoverflow.com/a/2304211/836450
	handle.setCallback(boost::bind(&PopSerial::characters_received,this,_1,_2));

	tx = this;
}

void PopSerial::init()
{
}


void PopSerial::characters_received(const char *data, size_t size)
{
	rx.process(data, size);
}


void PopSerial::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	handle.write(data, size);
}

}
