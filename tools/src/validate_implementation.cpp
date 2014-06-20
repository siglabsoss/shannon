#include <complex>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <dsp/common/poptypes.h>

#define BLACK "\033[22;30m"
#define RED "\033[22;31m"
#define GREEN "\033[22;32m"
#define YELLOW "\033[22;33m"
#define BLUE "\033[22;34m"
#define MAGENTA "\033[22;35m"
#define CYAN "\033[22;36m"
#define WHITE "\033[22;37m"
#define RESETCOLOR "\033[0m"

int check_type_compatibility(void)
{
	int r = 0;

	float test[2] = {2.0, 3.0};
	std::complex<float>* s = (std::complex<float>*)&test;
	cuComplex* c = (cuComplex*)&test;

	printf("Checking complex data type implementation compatibility ------------------------\r\n");

	printf("test[0]=%f, test[1]=%f\r\n", test[0], test[1]);

	printf("complex<float>->real()=%f, complex<float>->imag()=%f\r\n", s->real(), s->imag());

	printf("cuComplex->x=%f, cuComplex->y=%f\r\n", c->x, c->y);

	if( (s->real() != test[0]) || (s->imag() != test[1]) ||
	    (c->x != test[0]) || (c->y != test[1]) )
	{
		printf(RED "cuComplex and std::complex<float> data types are not equivalent!\r\n\r\n" RESETCOLOR);
		r = -1;
	}
	else
	{
		printf("single precision floating point complex "GREEN "pass\r\n\r\n" RESETCOLOR);
	}


	double testd[2] = {2.0, 3.0};
	std::complex<double>* d = (std::complex<double>*)&testd;
	popComplex* pc = (popComplex*)&testd;

	printf("test[0]=%f, test[1]=%f\r\n", test[0], test[1]);

	printf("complex<float>->real()=%f, complex<float>->imag()=%f\r\n", s->real(), s->imag());

	printf("popComplex->re=%f, popComplex->im=%f\r\n", pc->re, pc->im);

	if( (d->real() != test[0]) || (d->imag() != test[1]) ||
	    (pc->re != test[0]) || (pc->im != test[1]) )
	{
		printf(RED "popComplex and std::complex<double> data types are not equivalent!\r\n\r\n" RESETCOLOR);
		r = -1;
	}
	else
	{
		printf("double precision floating point complex "GREEN "pass\r\n\r\n" RESETCOLOR);
	}

	return r;
}

int main(int argc, char *argv[])
{
	int r = 0;
	printf("Validating implementation ------------------------------------------------------\r\n\r\n");

	r &= check_type_compatibility();

	return r;
}
