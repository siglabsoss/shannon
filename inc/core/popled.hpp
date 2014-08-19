/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_LED__
#define __POP_LED__

#include <stdint.h>

#include "poptimestamp.hpp"


namespace pop
{

enum LED_STATUS_T
{
	LED_STATUS_OK = 0,        // 0th bit
	LED_STATUS_PPS_ERROR,     // 1st bit
	LED_STATUS_WIFI_ERROR     // 2nd bit
};


class PopLED
{


public:
	PopLED();
	void flash(void);
	void poll(void);
	void set_error(LED_STATUS_T s);
	void clear_error(LED_STATUS_T s);

	PopTimestamp last_run;
	// time to wait between flashes
	double code_interval;
	bool print_code;

private:
	uint16_t codeword;
};

}

#endif
