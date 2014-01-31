#ifndef __POP_SERIAL_HPP_
#define __POP_SERIAL_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>

#include <string>     // string function definitions
#include <iostream>


#include <boost/asio.hpp>

using namespace std;



namespace pop
{


// http://www.webalice.it/fede.tft/serial_port/serial_port.html
// https://gitorious.org/serial-port/serial-port/source/03e161e0b788d593773b33006e01333946aa7e13:
class SimpleSerial
{


public:
    /**
     * Constructor.
     * \param port device name, example "/dev/tt yUSB0" or "COM4"
     * \param baud_rate communication speed, example 9600 or 115200
     * \throws boost::system::system_error if cannot open the
     * serial device
     */
    SimpleSerial(std::string port, unsigned int baud_rate)
    : io(), serial(io,port)
    {
        serial.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
    }

    /**
     * Write a string to the serial device.
     * \param s string to write
     * \throws boost::system::system_error on failure
     */
    void writeString(std::string s)
    {
        boost::asio::write(serial,boost::asio::buffer(s.c_str(),s.size()));
    }

    void write(const unsigned char *data, size_t size)
    {
    	boost::asio::write(serial,boost::asio::buffer(data,size));
    }

    void close()
    {
    	boost::system::error_code ec;
    	serial.close(ec);
    }

    /**
     * Blocks until a line is received from the serial device.
     * Eventual '\n' or '\r\n' characters at the end of the string are removed.
     * \return a string containing the received line
     * \throws boost::system::system_error on failure
     */
    std::string readLine()
    {
        //Reading data char by char, code is optimized for simplicity, not speed
        using namespace boost;
        char c;
        std::string result;
        for(;;)
        {
            asio::read(serial,asio::buffer(&c,1));
            switch(c)
            {
                case '\r':
                    break;
                case '\n':
                    return result;
                default:
                    result+=c;
            }
        }
    }

    unsigned char readChar()
	{
		//Reading data char by char, code is optimized for simplicity, not speed
		using namespace boost;
		unsigned c;

		asio::read(serial,asio::buffer(&c,1));
		return c;
	}

private:
    boost::asio::io_service io;
    boost::asio::serial_port serial;
};









class PopSerial : public PopSink<unsigned char>
{
public:
	 PopSink<unsigned char> *tx;
	 PopSource<unsigned char> rx; // serial receive generates characters
	 string path;
	 SimpleSerial handle;




	PopSerial(std::string devicePath, unsigned baud = 115200) : PopSink<unsigned char>("PopSerialSink", 1), rx("PopSerialSource"), path(devicePath), handle(devicePath, baud)
    {
		 tx = this;

		this->rx.set_loop_function(boost::bind(&PopSerial::run_loop,this));


		 handle.writeString("Serial Boot\r\n");

    }
    void init()
    {
    	while(1)
    	{
    		cout<<"Received : " << handle.readChar() << " : end" << endl;
    	}

//		handle.close();
    }

    // this gets called over and over in a while(1)
    unsigned int run_loop()
    {
    	cout << "run_loop" << endl;
//    	cout<<"Received : " << handle.readChar() << " : end" << endl;

    	return 0;
    }

    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {

    }
};

}


#endif
