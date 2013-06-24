/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <iostream>

#include <boost/date_time.hpp>
#include <boost/thread.hpp>

#include "popcontrol.hpp"

using namespace std;

namespace pop
{
	PopControl::PopControl()
	{

	}

	PopControl::~PopControl()
	{

	}

	int PopControl::run()
	{

		while(1)
		{
			// temporary pause for a few seconds
			boost::posix_time::seconds workTime(10);
			boost::this_thread::sleep(workTime);

			cout << "[PopControl] - keepalive" << endl;
		}
		
	}
}
