#ifndef __BCH_CODE_H__
#define __BCH_CODE_H__


#ifdef __cplusplus
extern "C" {
#endif

// all of your legacy C code here
typedef struct {
	int             m, n, length, k, t, d;
	int             p[21];
	int             alpha_to[1048576], index_of[1048576], g[548576];
	int             recd[1048576], data[1048576], bb[548576];
	int             seed;
	int             numerr, errpos[1024], decerror;
} bchcode_t;


void bch_init(bchcode_t* that, int mIN, int lengthIn);
void bch_compute_p(bchcode_t* that);
void bch_generate_gf(bchcode_t* that);

#ifdef __cplusplus
}
#endif



#endif
