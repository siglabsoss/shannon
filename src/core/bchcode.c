#include "core/bchcode.h"


#define TG that->g
#define TK that->k
#define TD that->d
#define TT that->t
#define TP that->p
#define TM that->m
#define TN that->n
#define TLENGTH that->length
#define TALPHA that->alpha_to
#define TINDEX that->index_of

void bch_init(bchcode_t* that, int mIN, int lengthIn, int tIN)
{
	int ninf;

	that->decerror = 0;

	TM = mIN;
	TLENGTH = lengthIn;
	TT = tIN;

	bch_compute_p(that);
	bch_generate_gf(that);
	bch_generate_polynomial(that);




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

void bch_generate_polynomial(bchcode_t* that)
/*
 * Compute the generator polynomial of a binary BCH code. Fist generate the
 * cycle sets modulo 2**m - 1, cycle[][] =  (i, 2*i, 4*i, ..., 2^l*i). Then
 * determine those cycle sets that contain integers in the set of (d-1)
 * consecutive integers {1..(d-1)}. The generator polynomial is calculated
 * as the product of linear factors of the form (x+alpha^i), for every i in
 * the above cycle sets.
 */
{
	register int	ii, jj, ll, kaux;
	register int	test, aux, nocycles, root, noterms, rdncy;
	int             cycle[1024][21];
	int             size[1024], min[1024];
	int             zeros[1024];


	/* Generate cycle sets modulo n, n = 2**m - 1 */
	cycle[0][0] = 0;
	size[0] = 1;
	cycle[1][0] = 1;
	size[1] = 1;
	jj = 1;			/* cycle set index */
//	if (m > 9)  {
//		printf("Computing cycle sets modulo %d\n", n);
//		printf("(This may take some time)...\n");
//	}
	do {
		/* Generate the jj-th cycle set */
		ii = 0;
		do {
			ii++;
			cycle[jj][ii] = (cycle[jj][ii - 1] * 2) % TN;
			size[jj]++;
			aux = (cycle[jj][ii] * 2) % TN;
		} while (aux != cycle[jj][0]);
		/* Next cycle set representative */
		ll = 0;
		do {
			ll++;
			test = 0;
			for (ii = 1; ((ii <= jj) && (!test)); ii++)
			/* Examine previous cycle sets */
			  for (kaux = 0; ((kaux < size[ii]) && (!test)); kaux++)
			     if (ll == cycle[ii][kaux])
			        test = 1;
		} while ((test) && (ll < (TN - 1)));
		if (!(test)) {
			jj++;	/* next cycle set index */
			cycle[jj][0] = ll;
			size[jj] = 1;
		}
	} while (ll < (TN - 1));
	nocycles = jj;		/* number of cycle sets modulo n */

//	printf("Enter the error correcting capability, t: ");
//	scanf("%d", &t);

	TD = 2 * TT + 1;

	/* Search for roots 1, 2, ..., d-1 in cycle sets */
	kaux = 0;
	rdncy = 0;
	for (ii = 1; ii <= nocycles; ii++) {
		min[kaux] = 0;
		test = 0;
		for (jj = 0; ((jj < size[ii]) && (!test)); jj++)
			for (root = 1; ((root < TD) && (!test)); root++)
				if (root == cycle[ii][jj])  {
					test = 1;
					min[kaux] = ii;
				}
		if (min[kaux]) {
			rdncy += size[min[kaux]];
			kaux++;
		}
	}
	noterms = kaux;
	kaux = 1;
	for (ii = 0; ii < noterms; ii++)
		for (jj = 0; jj < size[min[ii]]; jj++) {
			zeros[kaux] = cycle[min[ii]][jj];
			kaux++;
		}

	TK = TLENGTH - rdncy;

    if (TK<0)
      {
         printf("Parameters invalid!\n");
         exit(0);
      }
	printf("This is a (%d, %d, %d) binary BCH code\n", TLENGTH, TK, TD);

	/* Compute the generator polynomial */
	TG[0] = TALPHA[zeros[1]];
	TG[1] = 1;		/* g(x) = (X + zeros[1]) initially */
	for (ii = 2; ii <= rdncy; ii++) {
		TG[ii] = 1;
	  for (jj = ii - 1; jj > 0; jj--)
	    if (TG[jj] != 0)
	    	TG[jj] = TG[jj - 1] ^ TALPHA[(TINDEX[TG[jj]] + zeros[ii]) % TN];
	    else
	    	TG[jj] = TG[jj - 1];
	  TG[0] = TALPHA[(TINDEX[TG[0]] + zeros[ii]) % TN];
	}
	printf("Generator polynomial:\ng(x) = ");
	for (ii = 0; ii <= rdncy; ii++) {
	  printf("%d", TG[ii]);
	  if (ii && ((ii % 50) == 0))
	    printf("\n");
	}
	printf("\n");
}



