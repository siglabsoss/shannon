#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cufft.h>
#include <cuda/helper_cuda.h>

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
	if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, _cudaGetErrorEnum(err));
		cudaDeviceReset(); assert(0);
	}
}

//#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
//
//template<typename T>
//void check(T err, const char* const func, const char* const file, const int line) {
//  if (err != cudaSuccess) {
//    std::cerr << " ---------------------CUDA ERROR---------------------" << std::endl;
//    std::cerr << "Error at: " << file << ":" << line << std::endl;
//    std::cerr << "--> " << cudaGetErrorString(err) << " " << func << std::endl;
//    exit(EXIT_FAILURE);
//  }
//}




#endif
