/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>
#include <zmq.hpp>
#include <string>
#include <iostream>
#include <unistd.h>
#include "core/config.hpp"
#include "core/popserial.hpp"
#include "core/popgpsdevice.hpp"
#include "core/popartemisrpc.hpp"
#include "core/pops3prpc.hpp"
#include "core/popparsegps.hpp"
#include "core/poppackethandler.hpp"
#include "core/popchannelmap.hpp"
#include "core/popfabric.hpp"
#include "core/ldpc.hpp"
#include "core/popfabricbridge.hpp"
#include "core/utilities.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;



	Config::loadFromDisk();

	LDPC ldpc;


	ldpc.run();





    return 0;
}
