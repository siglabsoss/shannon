#ifndef __OBJECT_STASH_HPP_
#define __OBJECT_STASH_HPP_

#include <boost/unordered_map.hpp>
#include <core/popassert.h>
#include "popradio.h"

namespace pop
{

class ObjectStash
{
private:
	int quantity; // Number of storage spaces
	int next; // Next empty space
	// Pointer storage:
	void** storage;
	boost::unordered_map<unsigned long, PopRadio*> storage2;
	void inflate(int increase);
public:
	ObjectStash() : quantity(0), next(0), storage(0) {}
	~ObjectStash();
	int add(void* element);
	PopRadio* operator[](unsigned long key); // Fetch
	// Remove the reference from this PStash:
	void* remove(int index);
	// Number of elements in Stash:
	int count() const { return next; }

	PopRadio* findOrCreate(unsigned long key);
};



} // namespace pop

#endif
