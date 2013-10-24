#include "core/bchcode.h"


#define TP that->p
#define TM that->m
#define TN that->n
#define TLENGTH that->length
#define TALPHA that->alpha_to
#define TINDEX that->index_of

void bch_init(bchcode_t* that, int mIN, int lengthIn)
{
	int ninf;

	that->decerror = 0;

	TM = mIN;
	TLENGTH = lengthIn;

	bch_compute_p(that);
	bch_generate_gf(that);




	ninf = (TN + 1) / 2 - 1;

	if( !((TLENGTH <= TN)&&(TLENGTH>ninf)) )
	{
		printf("Error, code length of %d is not valid for this condition (%d < length <= %d)\n", TLENGTH, ninf, TN);
	}

}

/*
 * Builds lookup table P and computes N
 */
void bch_compute_p(bchcode_t* that)
{
	int i;

	for (i=1; i<TM; i++)
		TP[i] = 0;
	TP[0] = TP[TM] = 1;

	if (TM == 2)			TP[1] = 1;
	else if (TM == 3)	TP[1] = 1;
	else if (TM == 4)	TP[1] = 1;
	else if (TM == 5)	TP[2] = 1;
	else if (TM == 6)	TP[1] = 1;
	else if (TM == 7)	TP[1] = 1;
	else if (TM == 8)	TP[4] = TP[5] = TP[6] = 1;
	else if (TM == 9)	TP[4] = 1;
	else if (TM == 10)	TP[3] = 1;
	else if (TM == 11)	TP[2] = 1;
	else if (TM == 12)	TP[3] = TP[4] = TP[7] = 1;
	else if (TM == 13)	TP[1] = TP[3] = TP[4] = 1;
	else if (TM == 14)	TP[1] = TP[11] = TP[12] = 1;
	else if (TM == 15)	TP[1] = 1;
	else if (TM == 16)	TP[2] = TP[3] = TP[5] = 1;
	else if (TM == 17)	TP[3] = 1;
	else if (TM == 18)	TP[7] = 1;
	else if (TM == 19)	TP[1] = TP[5] = TP[6] = 1;
	else if (TM == 20)	TP[3] = 1;


	TN = 1;
	for (i = 0; i <= TM; i++) {
		TN *= 2;
//		printf("%1d", TP[i]);
	}
//	printf("\n");
	TN = TN / 2 - 1;
//	ninf = (TN + 1) / 2 - 1;

}


void bch_generate_gf(bchcode_t* that)
/*
 * Generate field GF(2**m) from the irreducible polynomial p(X) with
 * coefficients in p[0]..p[m].
 *
 * Lookup tables:
 *   index->polynomial form: alpha_to[] contains j=alpha^i;
 *   polynomial form -> index form:	index_of[j=alpha^i] = i
 *
 * alpha=2 is the primitive element of GF(2**m)
 */
{
	int    i, mask;

	mask = 1;
	TALPHA[TM] = 0;
	for (i = 0; i < TM; i++) {
		TALPHA[i] = mask;
		TINDEX[TALPHA[i]] = i;
		if (TP[i] != 0)
			TALPHA[TM] ^= mask;
		mask <<= 1;
	}
	TINDEX[TALPHA[TM]] = TM;
	mask >>= 1;
	for (i = TM + 1; i < TN; i++) {
		if (TALPHA[i - 1] >= mask)
			TALPHA[i] = TALPHA[TM] ^ ((TALPHA[i - 1] ^ mask) << 1);
		else
			TALPHA[i] = TALPHA[i - 1] << 1;
		TINDEX[TALPHA[i]] = i;
	}
	TINDEX[0] = -1;
}




