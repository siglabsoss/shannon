
// This define includes a main into our program for us
#define BOOST_TEST_DYN_LINK

// see http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/ for more info
#define BOOST_TEST_MODULE PopTests


//#include <boost/test/framework.hpp>
//#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
//#include <boost/test/unit_test_suite.hpp>


//Include any Pop stuff we are testing
#include <core/objectstash.hpp>
#include <core/popassert.h>
#include <core/config.hpp>

// include raw cpp files
#include <dsp/prota/popdeconvolve.cpp>
#include <core/objectstash.cpp>
#include <mdl/popradio.cpp>
#include <mdl/poptimestamp.cpp>
#include <core/config.cpp>

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








BOOST_AUTO_TEST_SUITE( timestamp_suite )

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

	PopTestSinkOne() : PopSink<PopTestMsg>("PopTestSinkOne", 0), verbose(0) { }
	void init() { }
	void process(const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		if(verbose) printf("received %lu PopMsg(s)\r\n", size);

		//        for( size_t i = 0; i < size; i++ )
		//        {
		//        	printf("Data was '%s'\r\n", (data+i)->origin);
		//        }

		if(verbose) printf("received %lu timestamps(s)\r\n", timestamp_size);
		for( size_t i = 0; i < timestamp_size; i++ )
		{
			if(verbose)
			{
				cout << "offset [" << timestamp_data[i].offset << "]" << endl;
				std::cout << "time was " << timestamp_data[i].get_full_secs() << std::endl;
				std::cout << "frac was " << timestamp_data[i].get_frac_secs() << std::endl;
			}
		}

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
    		t[0].offset = 0;
    		t[1] = PopTimestamp(4.0);
    		t[1].offset = chunk-1;


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
    		t[0].offset = 0;
    	}


    	for( size_t j = 1; j < stamps; j++ )
    	{
    		t[j] = PopTimestamp(t[0].get_real_secs()+ ((double)j/time_inc_divisor) );
    		t[j].offset = j;
    	}


    	process(b, count, t, stamps);

    }

    void send_manual_offset(size_t offset, double start_time = -1)
    {
    	// this whole function is copypasta
    	size_t count = 1;
    	size_t stamps = 1;

    	PopTestMsg b[count];
    	PopTimestamp t[stamps];

    	// build msgs
    	for( size_t i = 0; i < count; i++ )
    	{
    		char buff[20];
    		sprintf(buff, "Bob #%ld", i);
    		strcpy(b[i].origin, buff);
    	}


    	double time_inc_divisor = 100000.0;

    	// what to use for a start time
    	if( start_time != -1 )
    	{
    		t[0] = PopTimestamp(start_time);
    	}
    	else
    	{
    		// first timestamp is based on now
    		t[0] = PopTimestamp::get_system_time();
    		t[0].offset = offset;
    	}


    	for( size_t j = 1; j < stamps; j++ )
    	{
    		t[j] = PopTimestamp(t[0].get_real_secs()+ ((double)j/time_inc_divisor) );
    	}

    	process(b, count, t, stamps);

    }

};



BOOST_AUTO_TEST_CASE( timestamp_source_with_0_sample_size )
{
	PopTestSourceOne source;
	PopTestSinkOne sink;

	source.connect(sink);
	source.send_both(5, 5, 0, 10);

//	source.debug_print_timestamp_buffer();


	source.send_manual_offset(0);
//	source.debug_print_timestamp_buffer();


	source.send_both(5, 5, 1, 10);

//	source.debug_print_timestamp_buffer();

	BOOST_CHECK( source.timestamp_offsets_in_order(11) );
}


void testTwo(PopSink<PopTestMsg>* that, const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	static size_t index = 0;

	// 5 rows b/c we being called every two samples with 10 total

	// offset of first timestamp (forced), number timestamps, time,
	double result[5][3] =
	{
	{ 0,  1, 0.0},
	{ 1,  1, 0.3, },
	{ -1, 0, 0.3, },
	{ 0,  2, 0.6, },
	{ 1,  1, 0.9, }
	};

	double *test = result[index];

	BOOST_CHECK_EQUAL(test[0], that->calc_timestamp_offset(timestamp_data[0].offset) );
	BOOST_CHECK_EQUAL(test[1], timestamp_size );
	BOOST_CHECK_EQUAL(test[2], timestamp_data[0].get_frac_secs() );

	// bump index for next time we are called
	index++;
}

void testThree(PopSink<PopTestMsg>* that, const PopTestMsg* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
{
	static size_t index = 0;

	// 3 rows b/c we being called every three samples with 10 total (last sample is never sent to us)

	// offset of first timestamp (forced), number timestamps, time,
	double result[3][3] =
	{
	{ 0,  1, 0.0},
	{ 0,  1, 0.3, },
	{ 0,  2, 0.6, },
	};

	double *test = result[index];

	BOOST_CHECK_EQUAL(test[0], that->calc_timestamp_offset(timestamp_data[0].offset) );
	BOOST_CHECK_EQUAL(test[1], timestamp_size );
	BOOST_CHECK_EQUAL(test[2], timestamp_data[0].get_frac_secs() );

	// bump index for next time we are called
	index++;
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

// test the timestamp_offsets_in_order()
BOOST_AUTO_TEST_CASE( timestamp_timestamp_offsets_in_order )
{
	// note that we just have a source with no sink
	PopTestSourceOne source;

	// put in 5 samples with offset starting from 0
	source.send_both(5, 5);
	BOOST_CHECK( source.timestamp_offsets_in_order(5) );

	// put in another with offset starting from 0
	source.send_manual_offset(0);
	BOOST_CHECK( source.timestamp_offsets_in_order(6) );

	// put in another 5
	source.send_both(5, 5);
	BOOST_CHECK( source.timestamp_offsets_in_order(11) );

	// put in a bag egg
	// the reason this is bad is because we added 1 sample, but gave an offset of 1 which points at the second sample (which we didn't insert)
	source.send_manual_offset(1);
//	source.debug_print_timestamp_buffer();

	// put in another another 5 to make the 'bad egg' bad. This will cause an error because we give an offset of 0, which makes the previous out of bounds insertion clear
	source.send_both(5, 5);
	BOOST_CHECK(! source.timestamp_offsets_in_order(16) );

	//	source.debug_print_timestamp_buffer();

	// but things still look good if we only look back for 5 samples
	BOOST_CHECK(source.timestamp_offsets_in_order(5) );
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


#endif // UNIT_TEST
