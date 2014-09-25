/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <boost/thread/thread.hpp>

#include "core/popled.hpp"
#include "core/utilities.hpp"


namespace pop
{

using namespace std;


PopLED::PopLED() : code_interval(5), print_code(1), codeword(0)
{
	last_run = get_microsec_system_time();
}



void PopLED::poll(void)
{
	PopTimestamp now = get_microsec_system_time();
	now -= last_run;

	if( now.get_real_secs() <= code_interval )
	{
		return;
	}

	string silent(" > /dev/null 2>&1 ");

//	silent = "";

//	ostringstream os;

	// this should take about 0.5 seconds
	unsigned count = 200;

//	os << "sudo dd if=/dev/sda of=/dev/null bs=1024k count=" << count << " iflag=direct " << silent << " &";


	// handle "ok" bit
	if( codeword & 0xFFFE )
	{
		// if any flags are set, clear "ok"
		codeword &= ~0x0001;
	}
	else
	{
		// if no flags are set,
		codeword |= 0x0001;
	}

	if( print_code )
	{
		cout << "Error code: " << codeword;

		if( codeword == 1 )
		{
			cout << " (ok)";
		}

		cout << endl;
	}

//	for( uint16_t i = 0; i < codeword; i++ )
//	{
//		system(os.str().c_str());
//		boost::posix_time::milliseconds workTime(500);
//		boost::this_thread::sleep(workTime);
//		boost::this_thread::sleep(workTime);
//	}


	// update this
	last_run = get_microsec_system_time();
}

void PopLED::set_error(LED_STATUS_T s)
{
	codeword |= 0x01 << (unsigned) s;

}
void PopLED::clear_error(LED_STATUS_T s)
{
	codeword &= ~(0x01 << (unsigned) s);
}

}
