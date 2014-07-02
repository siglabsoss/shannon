/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_FABRIC__
#define __POP_FABRIC__

#include <stdint.h>
#include <time.h>
#include <zmq.hpp>
#include <iostream>
#include <unistd.h>
#include <sstream>

#include <map>
#include <string>


#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include "dsp/prota/popsparsecorrelate.h"


namespace pop
{

// Distributed In-memory store of basestation channel map
class PopFabric
{
public:
	PopFabric(zmq::context_t& context, std::string name, bool r, std::string ip_up);
	~PopFabric();

	unsigned poll();
	void send(std::string to, std::string message);
	void send_down(std::string to, std::string from, std::string message);
	void send_up(std::string to, std::string from, std::string message);
	void set_receive_function(boost::function<void(std::string, std::string)>);


private:
	unsigned router_poll();
	unsigned node_poll();
	boost::function<void(std::string, std::string)> fp;



	bool router;
	zmq::socket_t* pub_up;
	zmq::socket_t* sub_up;
	zmq::socket_t* pub_down;
	zmq::socket_t* sub_down;

	std::string name; // name of this node

	mutable boost::mutex mtx_;
};


}

#endif
