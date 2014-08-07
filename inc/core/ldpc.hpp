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

//FIXME: change to ints and dynamically allocate H http://www.eskimo.com/~scs/cclass/int/sx9b.html
#define h_rows (3)
#define h_cols (7)


#define LARGER_THAN_COLS (10)

namespace pop
{

typedef struct
{
	float llr;
} LDPC_N;

typedef struct
{
	float llr;
	unsigned degree;

	// index 0 is the smallest, 1 is the 2nd smallest
	float min[2];
	float min_index[2];

	// static defined for now
	unsigned node_index[LARGER_THAN_COLS];

	// 0 means that this check node is satisfied
	short parity;

} LDPC_M;



// Broadcast messaging fabric with to/from fields.  Upgradable to routed fabric
class LDPC
{
public:
	LDPC(short** H, unsigned u1, unsigned u2);
	~LDPC();

	void print_h();
	unsigned get_syndrome(void);
	void calc_syndrome(void);
	void run();
private:

	void prep_once();
	unsigned check_equations(void);
	void iteration();


	short H[h_rows][h_cols];
	LDPC_N n[h_cols];
	LDPC_M m[h_rows];
};




}

#endif
