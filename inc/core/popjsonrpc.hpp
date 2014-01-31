#ifndef __POP_JSON_RPC_HPP_
#define __POP_JSON_RPC_HPP_


#include <core/popsink.hpp>
#include <core/popsource.hpp>

using namespace std;

namespace pop
{

class PopJsonRPC : public PopSink<unsigned char>
{
public:
//	 PopSink<unsigned char> *tx; // PopSink must be inherited b/c of virtual classes, so we fake out a pointer here
	 PopSource<unsigned char> rx; // A member for rx

private:
	// pointer to a single timestamp given to us in the previous call to process
//	const PopTimestamp* previous_timestamp;
//
//	// the number of data samples from the previous call to process
//	size_t previous_size;
public:
	PopJsonRPC(size_t chunk) : PopSink<unsigned char>("PopJsonRPCSink", 1)
    {
//		 tx = this;
    }
    void init() {}
    void process(const unsigned char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {
    	unsigned int i = 0;
    	while(i < size) {
    		cout << data[i];
    		i++;
    	}

    }
};

}


#endif
