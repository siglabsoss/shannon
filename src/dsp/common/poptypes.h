
#ifndef __TYPES_H_
#define __TYPES_H_

typedef struct popComplex
{
	union
	{
		double re;
		double x;
	};
	union
	{
		double im;
		double y;
	};
} popComplex;

#endif // __TYPES_H_
