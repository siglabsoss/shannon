/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/


#include <iostream>
#include <string>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include "core/ldpc.hpp"
#include "core/utilities.hpp"




int main(int argc, char *argv[])
{
	using namespace boost;
	using namespace pop;
	using namespace std;



	//Config::loadFromDisk();

	short array = 0;
	unsigned rows = 3;
	unsigned cols = 7;


	LDPC ldpc( (short**)&array, rows, cols);


	ldpc.run();





    return 0;
}
