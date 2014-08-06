/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __LDPC_HPP__
#define __LDPC_HPP__

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

// Broadcast messaging fabric with to/from fields.  Upgradable to routed fabric
class LDPC
{
public:
	LDPC(void);
	~LDPC();
//
//	unsigned poll();
//	unsigned poll_burst(unsigned max = 1000);
//	void send(std::string to, std::string message);
//	void send_down(std::string to, std::string from, std::string message);
//	void send_up(std::string to, std::string from, std::string message);
//	void set_receive_function(boost::function<void(std::string, std::string, std::string)>);
//	void add_name(std::string name);
//	void keepalive();

	void run();
private:

//	unsigned poll_upwards();
//	boost::function<void(std::string, std::string, std::string)> fp;



	bool router;

//	std::vector<std::string> names; // name(s) of this node
//
//	mutable boost::mutex mtx_;
};


}

#endif
