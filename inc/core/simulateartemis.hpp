#ifndef __POP_SIM_ARTEMIS_HPP_
#define __POP_SIM_ARTEMIS_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>

#include <string>     // string function definitions
#include <iostream>
#include <queue>


#include <boost/asio.hpp>

using namespace std;



namespace pop
{


class SimulateArtemis : public PopSink<unsigned char>
{
public:
//	 PopSink<unsigned char> *tx;
	 PopSource<unsigned char> rx; // serial receive generates characters
	 std::queue<unsigned char> vtx;
//	 vector<char> vrx;
	 unsigned msPerChar;

	 void push_str(std::queue<unsigned char> &q, const char *s)
	 {
		 unsigned i = 0;
		 unsigned len = strlen(s);
		 for( i = 0; i < len; i++ )
		 {
			 q.push(s[i]);
		 }
	 }



	SimulateArtemis(unsigned msPerChar = 1000) : PopSink<unsigned char>("PopSimArtSink", 1), rx("PopSimArtSource"), msPerChar(msPerChar)
    {
//		tx = this;

		// this ugly syntax is required if we want to use start_thread()
		this->rx.set_loop_function(boost::bind(&SimulateArtemis::run_loop,this));

		vtx.push('\0');

		char str[] = "{\"method\":\"log\",\"params\":[\"hello printed\"],\"id\":3}";
		unsigned i = 0;
		unsigned len = strlen(str);
		for( i = 0; i < len; i++ )
		{
			vtx.push(str[i]);
		}

		vtx.push('\0');
		vtx.push('\0');
		vtx.push('\0');
		vtx.push('\0');

		push_str(vtx, "{\"method\":\"count\",\"params\":[],\"id\":4}");

		vtx.push('\0');

//		vtx.push('h');
//		vtx.push('e');
//		vtx.push('l');

    }
    void init()
    {
//		vtx
    }

    // this gets called over and over in a while(1), this function should not contain any sort of infinite loop itself
    // return non-zero to exit loop
    unsigned int run_loop()
    {


    	if( vtx.size() )
    	{
    		unsigned char* buf = rx.get_buffer(1);
    		buf[0] = this->vtx.front();
//    		cout<<"Char : " << buf[0] << " : end" << endl;
    		vtx.pop();
    		rx.process(1);
    	}
    	else
    	{
    		exit(0);
    	}

    	boost::this_thread::sleep(boost::posix_time::milliseconds(msPerChar));

    	return 0;
    }

    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {

    }
};

}


#endif
