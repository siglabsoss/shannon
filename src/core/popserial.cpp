#include "core/popserial.hpp"

using namespace std;

namespace pop
{

// how many characters to hold internally in-case signal processing down the line is running slowly
#define POP_SERIAL_SOURCE_SIZE (1250000)

PopSerial::PopSerial(std::string devicePath, unsigned baud, const char* name, bool p) : PopSink<char>(name, 0), rx("PopSerialSource", POP_SERIAL_SOURCE_SIZE), path(devicePath), print_all(p), handle(devicePath, baud)
{
	// note the weird _1 and _2. see http://stackoverflow.com/a/2304211/836450
	handle.setCallback(boost::bind(&PopSerial::characters_received,this,_1,_2));

	tx = this;

	//rx.debug_free_buffers = true;
}

PopSerial::~PopSerial()
{
	cout << "destroying PopSerial: " << this->get_name() << endl;
	handle.close();
	handle.clearCallback();
}


void PopSerial::init()
{
}


void PopSerial::characters_received(const char *data, size_t size)
{
	if( print_all )
	{
		cout << std::string(data, size) << endl;
	}
	rx.process(data, size);
}


void PopSerial::process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	handle.write(data, size);
}

}
