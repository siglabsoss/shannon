#include "core/objectstash.hpp"
#include <iostream>
#include <cstring> // 'mem' functions
using namespace std;

// these macros are shortcuts to the crazy long types that the map uses
#define _MAPTYPE boost::unordered_map<unsigned long, PopRadio*>
#define _ITERATOR _MAPTYPE::iterator
#define _VALUETYPE _MAPTYPE::value_type

namespace pop
{


bool ObjectStash::empty() const
{
	return storage2.empty();
}
unsigned long ObjectStash::size() const
{
	return storage2.size();
}

PopRadio* ObjectStash::findOrCreate(unsigned long key)
{
	_ITERATOR i = storage2.find(key);

	PopRadio* r;

	// if not found
	if( i == storage2.end() )
	{
//		printf("not found, creating\n");

		r = new PopRadio();
		r->setSerial(key);


		// simple way to insert
		//		storage2[key] = r;

		// fucking complicated way to insert:
		std::pair<_ITERATOR, bool> result = storage2.insert( _VALUETYPE(key, r) );

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
	_ITERATOR i = storage2.find(key);

	// if not found
	if( i == storage2.end() )
		return NULL;

	return i->second;
}




// No ownership:
ObjectStash::~ObjectStash() {
//  for(int i = 0; i < next; i++)
//	  PopAssertMessage(storage[i] == 0,
//      "ObjectStash not cleaned up");
//  delete []storage;
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

  _ITERATOR found = storage2.find(key);

  // key wasn't in the map
  if( found == storage2.end() )
  		return false;

  // remove from map
  storage2.erase(found);

  // delete from memory
  PopRadio* r = found->second;
  delete r;

  return true;
}


} //namespace
