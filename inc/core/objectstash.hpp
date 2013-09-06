#ifndef __OBJECT_STASH_HPP_
#define __OBJECT_STASH_HPP_

#include <boost/unordered_map.hpp>
#include <core/popassert.h>
#include "popradio.h"

namespace pop
{

class ObjectStash
{

public:
//	ObjectStash();
	~ObjectStash();

	PopRadio* operator[](unsigned long key); // same as findOrCreate
	bool remove(unsigned long key);
	unsigned long size() const;
	bool empty() const;

	PopRadio* findOrCreate(unsigned long key);
	PopRadio* find(unsigned long key);

private:
	boost::unordered_map<unsigned long, PopRadio*> storage;
};



} // namespace pop

#endif
