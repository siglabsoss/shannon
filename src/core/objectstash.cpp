// This file does memory management for a large number of PopRadio objects
// The core is the boost unordered_map (which is supposed to have better random access performance than map).
// The map contains pointers to PopObjects and calls new and delete at all the right times.
// The best way to use this is to just use the [] operator which short for findOrCreate().  See unit tests for examples
// This file could easily upgraded to use <T>emplates

#include "core/objectstash.hpp"
#include <boost/foreach.hpp>


using namespace std;


// these macros are shortcuts to the crazy long types that the map uses
#define _MAPTYPE boost::unordered_map<unsigned long, PopRadio*>
#define _ITERATOR _MAPTYPE::iterator
#define _VALUETYPE _MAPTYPE::value_type


namespace pop
{


bool ObjectStash::empty() const
{
	return storage.empty();
}
unsigned long ObjectStash::size() const
{
	return storage.size();
}

PopRadio* ObjectStash::findOrCreate(unsigned long key)
{
	_ITERATOR i = storage.find(key);

	PopRadio* r;

	// if not found
	if( i == storage.end() )
	{
//		printf("not found, creating\n");

		r = new PopRadio();
		r->setSerial(key);


		// simple way to insert
		//		storage2[key] = r;

		// fucking complicated way to insert:
		std::pair<_ITERATOR, bool> result = storage.insert( _VALUETYPE(key, r) );

		// split out tuple
//		_ITERATOR insertLocation = result.first;
		bool insertSuccess = result.second;

		// if we ever get failures here, I bet there's a concurrency issue
		if( !insertSuccess )
			printf("insert failed, possibly because key exists, however we just checked that!");

//		printf("Inserted at %x with success %d\r\n", insertLocation, (int)insertSuccess);
	}
	else
	{
		// if object was found

		// I couldn't find docs anywhere, but it appears that boost::iterator types dereference to a std::pair<key,value>
		// set r to the found object
		r = i->second;
	}

	return r;
}

// returns null if not found
PopRadio* ObjectStash::find(unsigned long key)
{
	_ITERATOR i = storage.find(key);

	// if not found
	if( i == storage.end() )
		return NULL;

	return i->second;
}




// the destructor deletes and empties the map
ObjectStash::~ObjectStash()
{

//	std::cout << "You were too lazy to clean up " << storage.size() << " items!!" << endl;

	// loop through storage and call delete
	// note that we don't modify the storage because this will mess up the loop
	BOOST_FOREACH( _VALUETYPE i, storage )
	{
//	    std::cout << "key is " << i.first << " value is " << i.second << std::endl;
		delete i.second;
	}

	// for an example of accessing an iterator inside loop:
	// http://stackoverflow.com/a/1858516/836450


	 // at this point storage hasn't changed, but now it just contains bad pointers
	 // so clear all the entries
	 storage.clear();
}

// Operator overloading syntax sugar for findOrCreate
PopRadio* ObjectStash::operator[](unsigned long key)
{
  return findOrCreate(key);
}

// removes object at key from map and calls delete
// returns success?  (true if object was found, false if object wasn't in there)
bool ObjectStash::remove(unsigned long key)
{

  _ITERATOR found = storage.find(key);

  // key wasn't in the map
  if( found == storage.end() )
  		return false;

  // remove from map
  storage.erase(found);

  // delete from memory
  PopRadio* r = found->second;
  delete r;

  return true;
}


} //namespace
