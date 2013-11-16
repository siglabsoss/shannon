
// This define includes a main into our program for us
#define BOOST_TEST_DYN_LINK

// see http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/ for more info
#define BOOST_TEST_MODULE PopTests


//#include <boost/test/framework.hpp>
//#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/lexical_cast.hpp>
//#include <boost/test/unit_test_suite.hpp>


//Include any Pop stuff we are testing
#include <core/objectstash.hpp>
#include <core/popassert.h>
#include <core/config.hpp>
#include "examples/popexamples.hpp"
#include "core/popsourcegpu.hpp"
#include "core/popsinkgpu.hpp"
#include "core/popfpgasource.h"

// include raw cpp files
#include <core/config.cpp>
#include <dsp/prota/popdeconvolve.cpp>
#include "dsp/prota/popchanfilter.cpp"
#include <core/objectstash.cpp>
#include <core/basestationfreq.c>
#include <mdl/popradio.cpp>
#include <mdl/poptimestamp.cpp>
#include <mdl/popsymbol.cpp>
#include "dsp/common/poptypes.cuh"

#include <iostream>
#include <fstream>
#include <string>
using namespace std;


using namespace pop;
using namespace rbx;

#ifdef UNIT_TEST

BOOST_AUTO_TEST_SUITE( object_stash_suite )

BOOST_AUTO_TEST_CASE( basic )
{

	// checks if argument is true, will continue if it isn't
	BOOST_CHECK( true );

	// will stop running tests beyond this point if argument isn't true
	BOOST_REQUIRE( true );

}


BOOST_AUTO_TEST_CASE( findOrCreate_sameness_uniqueness )
{
	ObjectStash s;

	// create radio with serial 1
	PopRadio *r1 = s.findOrCreate(1);
	// set the lat
	r1->setLat(0.3);

	// verify that we got back the correct serial
	BOOST_CHECK_EQUAL( r1->getSerial(), 1 );

	// do it again, this time we will find the one we just did
	PopRadio *r2 = s.findOrCreate(1);
	BOOST_CHECK_EQUAL( r1, r2 );
	BOOST_CHECK_EQUAL( r2->getLat(), 0.3 );

	PopRadio *r3 = s[1];

	BOOST_CHECK_EQUAL( r1, r3 );

	// create serial 2
	PopRadio *r4 = s[2];

	// verify it's not the same
	BOOST_CHECK(r1 != r4);

	// verify that we got back the correct serial
	BOOST_CHECK_EQUAL( r4->getSerial(), 2 );

}

BOOST_AUTO_TEST_CASE( find_returns_null )
{
	ObjectStash s;
	PopRadio *found;
	bool success;

	found = s.find(1);

	BOOST_CHECK(found == NULL);

	// empty
	BOOST_CHECK(s.empty());
	BOOST_CHECK_EQUAL(0, s.size());

	// create it
	PopRadio *r1 = s.findOrCreate(1);

	// not empty
	BOOST_CHECK(!s.empty());
	BOOST_CHECK_EQUAL(1, s.size());

	// find it and check that it's the same
	found = s.find(1);
	BOOST_CHECK_EQUAL(r1, found);

	// remove it
	success = s.remove(1);
	BOOST_CHECK(success);

	// find it and check that it's gone
	found = s.find(1);
	BOOST_CHECK(found == NULL);

	// remove it again (should return false)
	success = s.remove(1);
	BOOST_CHECK(!success);
}

BOOST_AUTO_TEST_CASE( stash_destructor )
{
	// dynamically create stash
	ObjectStash* s = new ObjectStash();

	// create an object, verify it went in
	s->findOrCreate(1);
	BOOST_CHECK_EQUAL(1, s->size());

	// call destructor, which will automatically delete all leftover PopRadio objects and empty storage
	delete s;

	// this test relies on UNDEFINED BEHAVIOUR by calling methods on deleted memory
	// but it works if you dare      [\/] [',,,,'] [\/]
//	BOOST_CHECK_EQUAL(0, s->size());

}


BOOST_AUTO_TEST_SUITE_END()














struct PopTestMsg
{
    char origin[20];
};


// this sink is a template that allows for the specification of the BITE_SIZE (how many bytes it requests at a time)
template <int BITE_SIZE>
class PopTestSinkTwo : public PopSink<PopTestMsg>
{
public:

	void (*fp)(PopSink<PopTestMsg>*, const PopTestMsg*, size_t, const PopTimestamp*, size_t);

	PopTestSinkTwo() : PopSink<PopTestMsg>("PopTestSinkTwo", BITE_SIZE), fp(0) { }
    void init() { }


    void process(const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {
        fp(this, data, size, timestamp_data, timestamp_size);
    }

};


class PopTestSinkOne : public PopSink<PopTestMsg>
{
public:

	bool verbose;
	bool verboseVerbose;
	// members which hold onto the most recent values of these variables from the last time process is called
	const PopTestMsg* m_lastData;
	size_t m_lastSize;

	PopTestSinkOne() : PopSink<PopTestMsg>("PopTestSinkOne", 0), verbose(0), verboseVerbose(0), m_lastData(0), m_lastSize(0) { }
	void init() { }
	void process(const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		m_lastData = data;
		m_lastSize = size;

		if(verbose || verboseVerbose) printf("received %lu PopMsg(s)\r\n", size);

		if(verboseVerbose)
		{
			for( size_t i = 0; i < size; i++ )
			{
				printf("Data was '%s'\r\n", (data+i)->origin);
			}
		}

		if(verbose || verboseVerbose) printf("received %lu timestamps(s)\r\n", timestamp_size);
		for( size_t i = 0; i < timestamp_size; i++ )
		{
			if(verbose || verboseVerbose)
			{
//				cout << "offset [" << timestamp_data[i].offset << "]" << endl;
				std::cout << "time was " << timestamp_data[i].get_full_secs() << std::endl;
				std::cout << "frac was " << timestamp_data[i].get_frac_secs() << std::endl;
			}
		}

	}
	
	// returns a hash calculated from the data given to us by the most recent call to process()
	size_t get_last_hash()
	{
		const unsigned char* addr = reinterpret_cast<const unsigned char*>(m_lastData);
		return boost::hash_range(addr, addr + m_lastSize);
	}

};

class PopTestSinkThree : public PopSink<PopTestMsg>, public PopSource<PopTestMsg>
{
public:

	bool verbose;

	PopTestSinkThree() : PopSink<PopTestMsg>("PopTestSinkThree", 10), verbose(0) { }
	void init() { }
	void process(const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{




		PopTestMsg *out = get_buffer(size/2);

		PopTimestamp* timestampOut = get_timestamp_buffer(timestamp_size);

		// correction factor for timestamps
//		double factor = 0.5;

		// copy every other sample
		for(size_t i = 0; i < size/2; i++ )
		{
			memcpy(out+i, data+(i*2), sizeof(PopTestMsg));

//			cout << "copy " << i << " from " << i*2 << endl;
		}

		// copy every other timsetamp
		for(size_t i = 0; i < size/2; i++ )
		{
			memcpy(timestampOut+i, timestamp_data+(i*2), sizeof(PopTimestamp));

//			cout << "copy " << i << " from " << i*2 << endl;
		}


		// process data
		PopSource<PopTestMsg>::process(out, size/2, timestampOut, timestamp_size);

	}

};


class PopTestSourceOne : public PopSource<PopTestMsg>
{
public:
	PopTestSourceOne() : PopSource<PopTestMsg>("PopTestSourceOne") { }

    void send_message(const char* desc, void*, size_t bytes)
    {

    }
    void start()
    {
    	int chunk = 50;

    	int j = 0;
    	while(chunk--)
    	{
    		j++;
    		PopTestMsg b[chunk];

    		for( int i = 0; i < chunk; i++ )
    		{
    			char buff[20];
    			sprintf(buff, "Bob #%d", i);
    			strcpy(b[i].origin, buff);
    		}


    		PopTimestamp t[4];
    		t[0] = PopTimestamp(3.3);
//    		t[0].offset = 0;
    		t[1] = PopTimestamp(4.0);
    		t[2] = PopTimestamp(5.0);
    		t[3] = PopTimestamp(7.0);
//    		t[1].offset = chunk-1;


    		process(b, chunk, t, 2);

    	}
    }

    void send_both(size_t count, size_t stamps, double start_time = -1, double time_inc_divisor = -1)
    {

    	PopTestMsg b[count];
    	PopTimestamp t[stamps];

    	// build msgs
    	for( size_t i = 0; i < count; i++ )
    	{
    		char buff[20];
    		sprintf(buff, "Bob #%ld", i);
    		strcpy(b[i].origin, buff);
    	}



    	// how much to divide the loop iterator by, and then add to time
    	if( time_inc_divisor == -1 )
    		time_inc_divisor = 100000.0;

    	// what to use for a start time
    	if( start_time != -1 )
    	{
    		t[0] = PopTimestamp(start_time);
    	}
    	else
    	{
    		// first timestamp is based on now
    		t[0] = PopTimestamp::get_system_time();
    	}


    	for( size_t j = 1; j < stamps; j++ )
    	{
    		t[j] = PopTimestamp(t[0].get_real_secs()+ ((double)j/time_inc_divisor) );
    	}


    	process(b, count, t, stamps);

    }

//    void send_manual_offset(size_t offset, double start_time = -1)
//    {
//    	// this whole function is copypasta
//    	size_t count = 1;
//    	size_t stamps = 1;
//
//    	PopTestMsg b[count];
//    	PopTimestamp t[stamps];
//
//    	// build msgs
//    	for( size_t i = 0; i < count; i++ )
//    	{
//    		char buff[20];
//    		sprintf(buff, "Bob #%ld", i);
//    		strcpy(b[i].origin, buff);
//    	}
//
//
//    	double time_inc_divisor = 100000.0;
//
//    	// what to use for a start time
//    	if( start_time != -1 )
//    	{
//    		t[0] = PopTimestamp(start_time);
//    	}
//    	else
//    	{
//    		// first timestamp is based on now
//    		t[0] = PopTimestamp::get_system_time();
//    		t[0].offset = offset;
//    	}
//
//
//    	for( size_t j = 1; j < stamps; j++ )
//    	{
//    		t[j] = PopTimestamp(t[0].get_real_secs()+ ((double)j/time_inc_divisor) );
//    	}
//
//    	process(b, count, t, stamps);
//
//    }

};

BOOST_AUTO_TEST_SUITE( timestamp_suite )


void debug_print_timestamp(PopTimestamp half)
{
	cout << "real() = " << half.get_real_secs() << endl;
	cout << "whole " << half.get_full_secs() << " and frac " << half.get_frac_secs() << endl;
}

BOOST_AUTO_TEST_CASE( timestamp_plus_overloads )
{
	double tol = 0.00000001;

	PopTimestamp half(0.6);
	PopTimestamp halfAgain(0.6);

	// add using the timestamp += timestamp overload
	half += halfAgain;

	// check if full seconds are wrapped correctly in a +=
	BOOST_CHECK_EQUAL( half.get_full_secs(), 1 );

	// check that frac secs were correctly modded
	BOOST_CHECK_CLOSE( half.get_frac_secs() , 0.2, tol );

	// start with a different fraction
	PopTimestamp other(0.4);

	// add using the timestamp += double overload
	other += 0.8;

	// check if things wrap correctly
	BOOST_CHECK_EQUAL( other.get_full_secs(), 1 );
	BOOST_CHECK_CLOSE( other.get_frac_secs() , 0.2, tol );

	// check that the two timestamps are the same
	BOOST_CHECK_CLOSE( half.get_real_secs() , other.get_real_secs(), tol );

	// this fails due to lambda
//	BOOST_CHECK( half == other );
}


//BOOST_AUTO_TEST_CASE( timestamp_source_with_0_sample_size )
//{
//	PopTestSourceOne source;
//	PopTestSinkOne sink;
//
//	source.connect(sink);
//	source.send_both(5, 5, 0, 10);
//
////	source.debug_print_timestamp_buffer();
//
//
//	source.send_manual_offset(0);
////	source.debug_print_timestamp_buffer();
//
//
//	source.send_both(5, 5, 1, 10);
//
////	source.debug_print_timestamp_buffer();
//
//	BOOST_CHECK( source.timestamp_offsets_in_order(11) );
//}


void testTwo(PopSink<PopTestMsg>* that, const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
//	static size_t index = 0;
//
//	// 5 rows b/c we being called every two samples with 10 total
//
//	// offset of first timestamp (forced), number timestamps, time,
//	double result[5][3] =
//	{
//	{ 0,  1, 0.0},
//	{ 1,  1, 0.3, },
//	{ -1, 0, 0.3, },
//	{ 0,  2, 0.6, },
//	{ 1,  1, 0.9, }
//	};
//
//	double *test = result[index];
//
//	BOOST_CHECK_EQUAL(test[0], that->calc_timestamp_offset(timestamp_data[0].offset, timestamp_buffer_correction) );
//	BOOST_CHECK_EQUAL(test[1], timestamp_size );
//	BOOST_CHECK_EQUAL(test[2], timestamp_data[0].get_frac_secs() );
//
//	// bump index for next time we are called
//	index++;
}

void testThree(PopSink<PopTestMsg>* that, const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
//	static size_t index = 0;
//
//	// 3 rows b/c we being called every three samples with 10 total (last sample is never sent to us)
//
//	// offset of first timestamp (forced), number timestamps, time,
//	double result[3][3] =
//	{
//	{ 0,  1, 0.0},
//	{ 0,  1, 0.3, },
//	{ 0,  2, 0.6, },
//	};
//
//	double *test = result[index];
//
//	BOOST_CHECK_EQUAL(test[0], that->calc_timestamp_offset(timestamp_data[0].offset, timestamp_buffer_correction) );
//	BOOST_CHECK_EQUAL(test[1], timestamp_size );
//	BOOST_CHECK_EQUAL(test[2], timestamp_data[0].get_frac_secs() );
//
//	// bump index for next time we are called
//	index++;
}

BOOST_AUTO_TEST_CASE( timestamp_staggering )
{
	PopTestSourceOne source;

	// these sinks accept 2 and 3 samples at a time
	PopTestSinkTwo<2> biteTwo;
	PopTestSinkTwo<3> biteThree;



	biteTwo.fp = &testTwo;
	biteThree.fp = &testThree;

	// connect them to the same source
	source.connect(biteTwo);
	source.connect(biteThree);


	// these 5 lines connect through a decimation block and handle the same data
	PopTestSinkThree  decimatePassThrough;
	PopTestSinkOne    finalPrintOutput;
	finalPrintOutput.verboseVerbose = false;


	source.connect(decimatePassThrough);
	decimatePassThrough.connect(finalPrintOutput);




	// the following commands build the following data picture:
	// the numbers are samples and the bars are timestamp points
	// also the timestamps are spaced such that each sample time is 0.1 seconds
	//  0123456789
	//  |  |  || |

	source.send_both(3, 1, 0,   10);
	source.send_both(3, 1, 0.3, 10);
	source.send_both(2, 2, 0.6, 10);
	source.send_both(1, 0, 0,   10);
	source.send_both(1, 1, 0.9, 10);

//	for( int i = 0; i < 1000; i++ )
//	{
//		source.send_both(3, 1, i+0,   10);
//		source.send_both(3, 1, i+0.3, 10);
//		source.send_both(2, 2, i+0.6, 10);
//		source.send_both(1, 0, i+0,   10);
//		source.send_both(1, 1, i+0.9, 10);
//	}

//	source.debug_print_timestamp_buffer();




}


BOOST_AUTO_TEST_CASE( timestamp_divide_by_zero_with_no_time )
{
	PopTestSourceOne source;
	PopTestSinkOne sink;
	source.connect(sink);

	// send 1 sample with 0 timestamps and check for divide by zero crash (actually it's mod by zero in this case)
	source.send_both(1, 0);
}


//BOOST_AUTO_TEST_CASE( timestamp_basic )
//{
//	PopTimestamp t;
//
//	t.tv_sec = 100;
//	t.tv_nsec = 10349123;
//
//	cout << "seconds: " << t.tv_sec << " ns: " << t.tv_nsec << endl;
//
//}


BOOST_AUTO_TEST_SUITE_END()



BOOST_AUTO_TEST_SUITE( file_read_write_suite )


BOOST_AUTO_TEST_CASE( file_readback )
{
	char filename[] = "file_readback_test.raw";
	size_t test_object_count = 3;

	// sink -> source to disk
	PopTestSourceOne source;
	PopDumpToFile<PopTestMsg> fileSink(filename);
	fileSink.flush_immediately = true;  // required because we are reading the file off the disk so soon after writing it that it's contents don't exist yet
	source.connect(fileSink);
	source.send_both(test_object_count, 0);



	// source from disk -> sink
	PopReadFromFile<PopTestMsg> fileSource(filename);
	PopTestSinkOne sink;
	fileSource.connect(sink);
	fileSource.read(test_object_count);

	size_t hashThroughDisk = sink.get_last_hash();


	// source -> sink

	PopTestSourceOne sourceDirect;
	PopTestSinkOne   sinkDirect;
	sourceDirect.connect(sinkDirect);
	sourceDirect.send_both(test_object_count, 0);
	size_t hashDirect = sinkDirect.get_last_hash();

	BOOST_CHECK_EQUAL( hashThroughDisk, hashDirect );


	// different value hash
	sourceDirect.send_both(test_object_count + 1, 0);
	size_t differentHash = sinkDirect.get_last_hash();

	BOOST_CHECK( differentHash != hashThroughDisk );

	// check to see if we can get a good hash again this time
	sourceDirect.send_both(test_object_count, 0);
	size_t hashDirectTwo = sinkDirect.get_last_hash();

	BOOST_CHECK_EQUAL( hashThroughDisk, hashDirectTwo );

}



BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE( channel_math )
BOOST_AUTO_TEST_CASE( basic_channel_math )
{
	return; // disable to visually check

	cout << "fbin size: " <<  boost::lexical_cast<string>(  bsf_fbin_size()  )   << endl;
	cout << "bottom: " <<  boost::lexical_cast<string>(  bsf_fft_bottom_frequency()  )   << endl;
	cout << "top   : " <<  boost::lexical_cast<string>(  bsf_fft_top_frequency()  )   << endl;
	cout << "bins per channel: " <<  boost::lexical_cast<string>(  bsf_fbins_per_channel()  )   << endl;

	cout << endl;

	for( int i = 0; i < 50; i++ )
	{
		cout << "channel " << i <<  ": " <<  boost::lexical_cast<string>(  bsf_channel_frequency(i)  )   << endl;
		cout << "above fft " << i <<  ": " <<  boost::lexical_cast<string>(  bsf_channel_frequency_above_fft(i)  )   << endl;
		cout << "fbin center " << i <<  ": " <<  boost::lexical_cast<string>(  bsf_channel_fbin_center(i)  )   << endl;

		cout << "fbin low  " << i <<  ": " <<  boost::lexical_cast<string>(  bsf_channel_fbin_low_exact(i)  )   << endl;
		cout << "fbin high " << i <<  ": " <<  boost::lexical_cast<string>(  bsf_channel_fbin_high_exact(i)  )   << endl;
	}
}

BOOST_AUTO_TEST_CASE( test_channel_frequency )
{
	double channels[50] = {902382812.5, 902433593.75, 902484375, 902535156.25, 902585937.5, 902636718.75, 902687500, 902738281.25, 902789062.5, 902839843.75, 902890625, 902941406.25, 902992187.5, 903042968.75, 903093750, 903144531.25, 903195312.5, 903246093.75, 903296875, 903347656.25, 903398437.5, 903449218.75, 903500000, 903550781.25, 903601562.5, 903652343.75, 903703125, 903753906.25, 903804687.5, 903855468.75, 903906250, 903957031.25, 904007812.5, 904058593.75, 904109375, 904160156.25, 904210937.5, 904261718.75, 904312500, 904363281.25, 904414062.5, 904464843.75, 904515625, 904566406.25, 904617187.5, 904667968.75, 904718750, 904769531.25, 904820312.5, 904871093.75};

	// test channel frequency calculation against every pre-defined one from the wiki
	for( int i = 0; i < 50; i++ )
	{
		BOOST_CHECK_EQUAL( channels[i], bsf_channel_frequency(i) );
	}

	size_t difference;

	// test if the rounded versions are returning discreet integers
	for( int i = 0; i < 50; i++ )
	{
		difference = bsf_channel_fbin_high(i) - bsf_channel_fbin_low(i);
		BOOST_CHECK_EQUAL( difference, 1040 ); // this is a hard check, this could also be bsf_bins_per_channel()
	}
}
BOOST_AUTO_TEST_SUITE_END()


class PopTestSourceTwo : public PopSource<PopTestMsg[50]>
{
public:
	PopTestSourceTwo() : PopSource<PopTestMsg[50]>("PopTestSourceTwo Array") { }


    void send_both(size_t count, size_t stamps, double start_time = -1, double time_inc_divisor = -1)
    {

    	PopTestMsg (*buff)[50] = get_buffer(10);

    	process();

    }

};


BOOST_AUTO_TEST_SUITE( lump_random )

BOOST_AUTO_TEST_CASE( basic_array_sink_source )
{
	PopTestSourceTwo arraySource();

}

BOOST_AUTO_TEST_CASE( pak_basic )
{
	uint8_t storage[2];

//	pak_print(storage, 16);



	// set this to 0
	storage[0] = 0;
	storage[1] = 0;

	// set 1's up the byte
	uint8_t mask = 0xFE;
	uint8_t check;
	for( int i = 0; i < 8; i++ )
	{
		pak_change_bit(storage, i, 1);
//		pak_print(storage1, 8);

		check = 0xff & ~mask;

		BOOST_CHECK_EQUAL(check, storage[0]);

		mask <<=1;
	}

	// final check
	BOOST_CHECK_EQUAL(0xff, storage[0]);

	// verify no spillover
	BOOST_CHECK_EQUAL(0x00, storage[1]);

	// now check zeroing out
	storage[0] = 0xff;
	storage[1] = 0xff;

	mask = 0x7f;
	for( int i = 7; i >= 0; i-- )
	{
		pak_change_bit(storage, i, 0);
//		pak_print(storage1, 8);

		check = 0xff & mask;

		BOOST_CHECK_EQUAL(check, storage[0]);

		mask >>=1;
	}

	// final check
	BOOST_CHECK_EQUAL(0x00, storage[0]);

	// verify no spillover
	BOOST_CHECK_EQUAL(0xff, storage[1]);


}

BOOST_AUTO_TEST_CASE( basic_fpga )
{
	PopFPGASource source;

	source.init();

}


unsigned gpu_kernel(unsigned i, unsigned len, unsigned half_len, unsigned three_quarter, unsigned fbins)
{


	int sample;

	sample = (i/half_len) * len + ((i%half_len)+three_quarter) % len;

	int b = i % half_len; // fft sample out

	int fbin_out = i / half_len; // fbin out


//	cout << sample << " goes into ";
	printf("%u goes into [%u][%u]\r\n", sample, fbin_out, b);

	return 0;

}
BOOST_AUTO_TEST_CASE( gpu_math_logic_check )
{

	int spread_len = 4;

	int total = spread_len * SPREADING_BINS;

	int total2 = 1030;

	for( int i = 0; i < total2; i++ )
	{
//		gpu_kernel(i, spread_len*2, spread_len, (spread_len*6)/4, SPREADING_BINS);
	}



}

BOOST_AUTO_TEST_SUITE_END()

class PopTestGpuSourceOne : public PopSourceGpu<char>
{
public:
	PopTestGpuSourceOne() : PopSourceGpu<char>("PopTestGpuSourceOne", 2) { }


    void send_both(size_t count)
    {
    	static char c = 'a';
    	char h_buff[3];
    	h_buff[0] = c++;
    	h_buff[1] = c++;
//    	h_buff[2] = c++;

    	char* d_buff = get_buffer();

    	cudaMemcpy(d_buff, h_buff, 2 * sizeof(char), cudaMemcpyHostToDevice);


    	static double time = 0.333333333333333;
    	PopTimestamp h_ts[3];
    	h_ts[0] = PopTimestamp(time++);
    	h_ts[1] = PopTimestamp(time++);

    	PopTimestamp* d_ts = get_timestamp_buffer();

    	cudaMemcpy(d_ts, h_ts, 2 * sizeof(PopTimestamp), cudaMemcpyHostToDevice);



		// call to GPU process accepts no parameters because everything is fixed
    	process();
    }

};

class PopTestGpuSinkOne : public PopSinkGpu<char>
{
public:

	bool verbose;
	bool verboseVerbose;


	PopTestGpuSinkOne() : PopSinkGpu<char>("PopTestGpuSinkOne", 3), verbose(0), verboseVerbose(0) { }
	void init() { }
	void process(const char* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		char h_buff[3];
		PopTimestamp h_ts[3];

		cudaMemcpy(h_buff, data, 3 * sizeof(char), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ts, timestamp_data, 3 * sizeof(PopTimestamp), cudaMemcpyDeviceToHost);

//		m_lastData = data;
//		m_lastSize = size;
//
		if(verbose || verboseVerbose) printf("received %lu Data samples(s)\r\n", size);

		if(verboseVerbose)
		{
			for( size_t i = 0; i < size; i++ )
			{
				printf("Data was '%c'\r\n", h_buff[i]);
			}
		}

		if(verbose || verboseVerbose) printf("received %lu timestamps(s)\r\n", timestamp_size);
		for( size_t i = 0; i < timestamp_size; i++ )
		{
			if(verbose || verboseVerbose)
			{
				std::cout << h_ts[i] << std::endl;
			}
		}

	}

};



BOOST_AUTO_TEST_SUITE( sink_source_gpu )

BOOST_AUTO_TEST_CASE( basic_gpu_sink_source )
{
	PopTestGpuSourceOne source;
	PopTestGpuSinkOne sink;
//	sink.verboseVerbose = true;

	source.connect(sink);

//	cout << endl << "starting pos source_idx() = " << source.m_buf.source_idx() << " sink_idx() = " << source.m_buf.sink_idx(sink.m_sourceBufIdx) << endl;

//	source.debug_print();

	for( int i = 0; i < 15; i++ )
	{
		source.send_both(0);
//		cout << endl << "after call " << i << " source_idx() = " << source.m_buf.source_idx() << " sink_idx() = " << source.m_buf.sink_idx(sink.m_sourceBufIdx) << endl;

//		source.debug_print();
	}


}

BOOST_AUTO_TEST_SUITE_END()



#endif // UNIT_TEST
