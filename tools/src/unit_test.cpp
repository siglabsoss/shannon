#include <iostream>

#include <complex>
#include <cstdlib>

#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;

using namespace boost::posix_time;


#ifdef CFC_UNIT_TEST

int main(int argc, char *argv[])
{
	complex<float>* cfc;
	ptime t1, t2;
	time_duration td, tLast;

	cfc = (complex<float>*)malloc(512*2*sizeof(complex<float>)); ///< pad

	t1 = microsec_clock::local_time();
	gpu_gen_pn_match_filter_coef( pn, cfc, 512, 512, 0.5 );
	t2 = microsec_clock::local_time();

	td = t2 - t1;

	cout << "gen_pn_match_filter_coef() time = " << td.total_microseconds() << "us" << endl;

	free(cfc);

	return 0;
}

#endif
