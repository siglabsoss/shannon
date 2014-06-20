


#ifndef __POP_TYPES_CU__
#define __POP_TYPES_CU__

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


// sets or clears a bit in a packed uint8_t array
__host__ __device__ void pak_change_bit(uint8_t storage[], unsigned index, unsigned value)
{
	unsigned j, k;
	j = (int) index/8;
	k = index % 8;

	if( value )
		storage[j] |= 0x1 << k;
	else
		storage[j] &= ~(0x1 << k);
}

__host__ void pak_print(uint8_t storage[], unsigned count)
{
	unsigned j, k;
	uint8_t bit;

	j = count/8;

	cout << "hex: 0x";

	while(j--)
	{
		cout << hex << setfill('0') << setw(2) << (int)storage[j];
	}

	cout << endl;



	cout << "bits: " << endl;



	for(unsigned i = 0; i < count; i++)
	{
		j = (int) i/8;
		k = i % 8;

		bit = ( storage[j] >> k ) & 0x01;

		cout << (int) bit << endl;
	}
}

#endif
