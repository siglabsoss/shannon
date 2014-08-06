/******************************************************************************
* Copyright 2014 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <assert.h>
#include <stddef.h>

#include <vector>

#include <boost/thread/mutex.hpp>

#include "core/ldpc.hpp"
#include "frozen/frozen.h"
#include "core/utilities.hpp"
#include "core/poppackethandler.hpp"

using boost::mutex;
using std::make_pair;
using std::pair;
using std::string;
using std::vector;
using namespace zmq;
using namespace std;


namespace pop
{

//FIXME: add simple routing to not send messages back to sender during a send_down()

LDPC::LDPC(void) : router(false)
{

}


LDPC::~LDPC()
{


}

void LDPC::run()
{
	// setup H, our code matrix
	int h_rows = 3;
	int h_cols = 7;

	short H[h_rows][h_cols];

	// fill H with 0's
	memset(H, 0, sizeof(H[0][0]) * h_rows * h_cols);

	// set 1's

	H[0][0] = 1;
	H[0][1] = 1;
//	H[0][2] = 1;
	H[0][3] = 1;
//	H[0][4] = 1;
//	H[0][5] = 1;
//	H[0][6] = 1;

//	H[1][0] = 1;
//	H[1][1] = 1;
	H[1][2] = 1;
	H[1][3] = 1;
	H[1][4] = 1;
//	H[1][5] = 1;
//	H[1][6] = 1;

//	H[2][0] = 1;
//	H[2][1] = 1;
//	H[2][2] = 1;
	H[2][3] = 1;
//	H[2][4] = 1;
	H[2][5] = 1;
	H[2][6] = 1;


//	for( int i = 0; i < h_rows; ++i )
//	{
//		for( int j = 0; j < h_cols; ++j)
//		{
//			cout << H[i][j] << ",";
//		}
//		cout << endl;
//	}


	cout << "running"  << endl;
}


}
