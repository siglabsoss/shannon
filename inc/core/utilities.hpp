/*
 * This file is for small utilities and the like, not to be confused with dsp/utils.hpp which is for cuda utilies
 */

#ifndef __STANDARD_UTILITIES_HPP__
#define __STANDARD_UTILITIES_HPP__

// returns a random value between to floats, min, max.  run srand before
#define RAND_BETWEEN(Min,Max)  (((double(rand()) / double(RAND_MAX)) * (Max - Min)) + Min)

#endif
