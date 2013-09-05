#ifndef __OBJECT_STASH_HPP_
#define __OBJECT_STASH_HPP_

#include <core/popassert.h>

namespace pop
{

class ObjectStash
{
private:
	int quantity; // Number of storage spaces
	int next; // Next empty space
	// Pointer storage:
	void** storage;
	void inflate(int increase);
public:
	ObjectStash() : quantity(0), storage(0), next(0) {}
	~ObjectStash();
	int add(void* element);
	void* operator[](int index) const; // Fetch
	// Remove the reference from this PStash:
	void* remove(int index);
	// Number of elements in Stash:
	int count() const { return next; }
};



} // namespace pop

#endif
