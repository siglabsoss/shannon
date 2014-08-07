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
	// h_rows can never be less than 2 due to an optimization
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


	cout << "H = " << endl;
	for( int i = 0; i < h_rows; ++i )
	{
		for( int j = 0; j < h_cols; ++j)
		{
			cout << H[i][j] << ",";
		}
		cout << endl;
	}
	cout << endl;

	// Inboud message

	// negative represents a 1
	LDPC_N n[h_cols];
	n[0].llr = -9;
	n[1].llr = -7;
	n[2].llr = -12;
	n[3].llr = +4;
	n[4].llr = +7;
	n[5].llr = +10;
	n[6].llr = -11;

	LDPC_M m[h_rows];

	// calculate degree for each check node, fill in node_index
	for( int i = 0; i < h_rows; ++i )
	{
		m[i].degree = 0;
		for( int j = 0; j < h_cols; ++j)
		{
			if( H[i][j] )
			{
				// keep track of the index of which variable nodes this check node is connected to
				m[i].node_index[m[i].degree] = j;
				m[i].degree++;
			}
		}

		cout << "Check node " << i << " has degree " << m[i].degree << endl;
	}


	// iteration

	// Load up min 1,2
	for( int i = 0; i < h_rows; ++i )
	{
		LDPC_M *mi = &(m[i]);
		LDPC_N *ni = &(n[mi->node_index[0]]);
		LDPC_N *ni_next = &(n[mi->node_index[1]]);

		// cook first two iterations of loop below
		if( abs(ni->llr) < abs(ni_next->llr) )
		{
			mi->min[0]       = ni->llr;
			mi->min_index[0] = mi->node_index[0];

			mi->min[1]       = ni_next->llr;
			mi->min_index[1] = mi->node_index[1];
		}
		else
		{
			mi->min[1]       = ni->llr;
			mi->min_index[1] = mi->node_index[0];

			mi->min[0]       = ni_next->llr;
			mi->min_index[0] = mi->node_index[1];
		}

		// start at 2
		for( unsigned j = 2; j < mi->degree; ++j)
		{
			int node_index = mi->node_index[j];
			ni = &(n[node_index]);

			if( abs(ni->llr) < abs(mi->min[0]) )
			{
				// this is the smallest yet.  bump the previous smallest to 2nd smallest
				mi->min[1] = mi->min[0];
				mi->min_index[1] = mi->min_index[0];

				// and assign
				mi->min[0] = ni->llr;
				mi->min_index[0] = node_index;
			}
			else if( abs(ni->llr) < abs(mi->min[1]) )
			{
				// this is only smaller than the 2nd smallest, simply assign
				mi->min[1] = ni->llr;
				mi->min_index[1] = node_index;
			}
		}
	}


	cout << "running"  << endl;
}


}
