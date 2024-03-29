/*
 * This file is for small utilities and the like, not to be confused with dsp/utils.hpp which is for cuda utilies
 * CPP code should be put here, C code should be put in util.h
 */

#ifndef __STANDARD_UTILITIES_HPP__
#define __STANDARD_UTILITIES_HPP__

#include <sstream>
#include <string>

#include "poptimestamp.hpp"
#include "dsp/prota/popsparsecorrelate.h"

// returns a random value between to floats, min, max.  run srand before
#define RAND_BETWEEN(Min,Max)  (((double(rand()) / double(RAND_MAX)) * (Max - Min)) + Min)


namespace pop
{

uuid_t b64_to_uuid(std::string b64_serial);
std::string uuid_to_b64(uuid_t u);
bool operator==(const uuid_t& lhs, const uuid_t& rhs);
PopTimestamp get_microsec_system_time(void);
int getch(void);
int kbhit(void);
std::string pop_get_hostname(void);



template<typename T>
T parseNumber(const std::string& in)
{
	T result;
	std::stringstream ss(in);
	ss >> result;
	return result;
}


// This macro is sugar for the std string constructor when using the frozen json library
#define FROZEN_GET_STRING(token) std::string(token->ptr, token->len)


}

#endif
