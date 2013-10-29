
#include "poptypes.cuh"

__host__ __device__ double magnitude2( const popComplex& in )
{
	return in.re * in.re + in.im * in.im;
}

__host__ __device__ popComplex operator*(const popComplex& a, const popComplex& b)
{
	popComplex r;
	r.re = b.re*a.re - b.im*a.im;
	r.im = b.im*a.re + b.re*a.im;
	return r;
}

__host__ __device__ popComplex operator+(const popComplex& a, const popComplex& b)
{
	popComplex r;
	r.re = a.re + b.re;
	r.im = a.im + b.im;
	return r;
}

