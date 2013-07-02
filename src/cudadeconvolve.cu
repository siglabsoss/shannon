/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

//#include <cuComplex.h>
#include <complex>
#include <iostream>

using namespace std;

struct cuComplex
{
	float r;
	float i;

	__device__ cuComplex( float a, float b ) : r(a), i(b) {}

	__device__ float magnitude2( void )
	{
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}

	__device__ cuComplex operator+=(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}
};



__global__ void deconvolve(cuComplex *pn, cuComplex *data, 
	cuComplex *old_data, float *product)
{
	int i = threadIdx.x;
	int N = blockDim.x;
	int I = N - i;
	int n;
	cuComplex s = cuComplex(0.0, 0.0);

	for( n = 0; n < I; n++)
		s += data[n] * pn[n + i];
	for( n = i; n < N; n++)
		s += old_data[n] * pn[n + I];

	product[i] = s.magnitude2();
}

extern "C"
{	
	cudaError_t cu_err;
	cuComplex *_pcode;
	cuComplex *_data1;
	cuComplex *_data2;
	float *_prod1;
	size_t _len;
	int _buf_idx;

	void init_deconvolve(complex<float> *pn, size_t len)
	{
		_len = len;
		cudaMalloc(&_pcode, len * sizeof(cuComplex));
		cudaMalloc(&_data1, len * sizeof(cuComplex));
		cudaMalloc(&_data2, len * sizeof(cuComplex));
		cudaMalloc(&_prod1, len * sizeof(float));

		cudaMemset(&_data2, 0, len * sizeof(cuComplex));
		cudaMemcpy(_pcode, pn, len, cudaMemcpyHostToDevice);
	}

	void start_deconvolve(complex<float> *data, float *product)
	{
		if( 1 == _buf_idx ) _buf_idx = 0;
		else _buf_idx = 1;

		cudaMemcpy(_buf_idx?_data1:_data2, data, _len, cudaMemcpyHostToDevice);
		deconvolve<<<1, _len>>>(_pcode, _buf_idx?_data1:_data2, _buf_idx?_data2:_data1, _prod1);

		// check for errors... ie, allocating too many threads/block =P
		cu_err = cudaGetLastError();
	    if (cu_err != cudaSuccess)
	    {
	        std::cout << "CUDA_ERROR: deconvolve returned " << cu_err << " -> " << cudaGetErrorString(cu_err) << std::endl;
	        exit(EXIT_FAILURE);
	    }

		cudaMemcpy(product, _prod1, _len, cudaMemcpyDeviceToHost);
	}

}
