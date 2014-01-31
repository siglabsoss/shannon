#ifndef __POP_SERIAL_HPP_
#define __POP_SERIAL_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <string>     // string function definitions
#include <unistd.h>     // UNIX standard function definitions
#include <fcntl.h>      // File control definitions
#include <errno.h>      // Error number definitions
#include <termios.h>    // POSIX terminal control definitions
#include <iostream>


#include <boost/asio.hpp>

using namespace std;



namespace pop
{






using namespace boost;
// http://www.webalice.it/fede.tft/serial_port/serial_port.html
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

private:
    boost::asio::io_service io;
    boost::asio::serial_port serial;
};









class PopSerial : public PopSink<unsigned char>
{
public:
	 PopSink<unsigned char> *tx;
	 PopSource<unsigned char> rx;
	 string path;
	 int fd;


	PopSerial(std::string devicePath) : PopSink<unsigned char>("PopSerialSink", 1), rx("PopSerialSource"), path(devicePath)
    {
		 tx = this;

		 SimpleSerial serial("/dev/ttyO4",115200);

		 serial.writeString("Hello world\n");

//		 // open file on disk
//		 fd = open( path.c_str(), O_RDWR | O_NOCTTY );
//
//		 struct termios tty;
//		 struct termios tty_old;
//		 memset (&tty, 0, sizeof tty);
//
//		 /* Error Handling */
//		 if ( tcgetattr ( fd, &tty ) != 0 )
//		 {
//			 cout << "Error " << errno << " from tcgetattr: " << strerror(errno) << endl;
//		 }
//
//		 /* Save old tty parameters */
//		 tty_old = tty;
//
//		 /* Set Baud Rate */
//		 cfsetospeed (&tty, (speed_t)B115200);
//		 cfsetispeed (&tty, (speed_t)B115200);
//
//		 /* Setting other Port Stuff */
//		 tty.c_cflag     &=  ~PARENB;            // Make 8n1
//		 tty.c_cflag     &=  ~CSTOPB;
//		 tty.c_cflag     &=  ~CSIZE;
//		 tty.c_cflag     |=  CS8;
//
//		 tty.c_cflag     &=  ~CRTSCTS;           // no flow control
//		 tty.c_cc[VMIN]   =   1;                 // read doesn't block
//		 tty.c_cc[VTIME]  =   5;                 // 0.5 seconds read timeout
//		 tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines
//
//		 /* Make raw */
//		 cfmakeraw(&tty);
//
//		 /* Flush Port, then applies attributes */
//		 tcflush( fd, TCIFLUSH );
//		 if ( tcsetattr ( fd, TCSANOW, &tty ) != 0)
//		 {
//			 cout << "Error " << errno << " from tcsetattr" << endl;
//		 }


    }
    void init()
    {
    	cout << "init " << endl;
    }

    void writeUart(char bytes[], unsigned length) {

    	write( fd, bytes, length);
    }



    void readUart()
    {
    	int n = 3;
    	std::string response;
    	char buf [256];

    	do
    	{
			n = read( fd, &buf, 1 );
			response.append( buf );
    	}
    	while( n-- > 0);

//    	if (n < 0)
//    	{
//    		cout << "Error reading: " << strerror(errno) << endl;
//    	}
//    	else if (n == 0)
//    	{
//    	    cout << "Read nothing!" << endl;
//    	}
//    	else
//    	{
    	    cout << "Response: " << response;
//    	}
    }

    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {

    }
};

}


#endif
