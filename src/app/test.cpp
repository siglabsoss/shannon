
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

#include <iostream>
#include <fstream>
#include <string>
using namespace std;


using namespace pop;


BOOST_AUTO_TEST_SUITE( object_stash_suite )

BOOST_AUTO_TEST_CASE( basic )
{

	// checks if argument is true, will continue if it isn't
	BOOST_CHECK( true );

	// will stop running tests beyond this point if argument isn't true
	BOOST_REQUIRE( true );

}


// test taken from http://www.cs.ust.hk/~dekai/library/ECKEL_Bruce/TICPP-2nd-ed-Vol-one/TICPP-2nd-ed-Vol-one-html/Chapter13.html
//BOOST_AUTO_TEST_CASE( baisc_add_check )
//{
//	// this is broken for now
//	return;
//
////	 s = ObjectStash();
//
//	 ObjectStash intStash;
//	  // 'new' works with built-in types, too. Note
//	  // the "pseudo-constructor" syntax:
//	  for(int i = 0; i < 25; i++)
//	    intStash.add(new int(i));
//	  for(int j = 0; j < intStash.count(); j++)
//	    cout << "intStash[" << j << "] = "
//	         << *(int*)intStash[j] << endl;
//	  // Clean up:
//	  for(int k = 0; k < intStash.count(); k++)
//	    delete intStash.remove(k);
//
//	  ObjectStash stringStash;
//
//	  stringStash.add(new string("what"));
//	  stringStash.add(new string("is"));
//	  stringStash.add(new string("the"));
//	  stringStash.add(new string("deal"));
//	  stringStash.add(new string("with"));
//	  stringStash.add(new string("airplane"));
//	  stringStash.add(new string("food!?"));
//
//	  // Print out the strings:
//	  for(int u = 0; stringStash[u]; u++)
//	    cout << "stringStash[" << u << "] = "
//	         << *(string*)stringStash[u] << endl;
//	  // Clean up:
//	  for(int v = 0; v < stringStash.count(); v++)
//	    delete (string*)stringStash.remove(v);
//
//}

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


BOOST_AUTO_TEST_SUITE_END()
