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

int ObjectStash::add(void* element) {
  const int inflateSize = 10;
  if(next >= quantity)
    inflate(inflateSize);
  storage[next++] = element;
  return(next - 1); // Index number
}

PopRadio* ObjectStash::findOrCreate(unsigned long key) {
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


// No ownership:
ObjectStash::~ObjectStash() {
  for(int i = 0; i < next; i++)
	  PopAssertMessage(storage[i] == 0,
      "ObjectStash not cleaned up");
  delete []storage;
}

// Operator overloading syntax sugar for findOrCreate
PopRadio* ObjectStash::operator[](unsigned long key) {
  return findOrCreate(key);
}

void* ObjectStash::remove(int index) {
  void* v = operator[](index);
  // "Remove" the pointer:
  if(v != 0) storage[index] = 0;
  return v;
}

void ObjectStash::inflate(int increase) {
  const int psz = sizeof(void*);
  void** st = new void*[quantity + increase];
  memset(st, 0, (quantity + increase) * psz);
  memcpy(st, storage, quantity * psz);
  quantity += increase;
  delete []storage; // Old storage
  storage = st; // Point to new memory
} ///:~

} //namespace
