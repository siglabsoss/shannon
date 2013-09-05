#include "core/objectstash.hpp"
#include <iostream>
#include <cstring> // 'mem' functions
using namespace std;

namespace pop
{

int ObjectStash::add(void* element) {
  const int inflateSize = 10;
  if(next >= quantity)
    inflate(inflateSize);
  storage[next++] = element;
  return(next - 1); // Index number
}

// No ownership:
ObjectStash::~ObjectStash() {
  for(int i = 0; i < next; i++)
	  PopAssertMessage(storage[i] == 0,
      "ObjectStash not cleaned up");
  delete []storage;
}

// Operator overloading replacement for fetch
void* ObjectStash::operator[](int index) const {
	PopAssertMessage(index >= 0,
    "ObjectStash::operator[] index negative");
  if(index >= next)
    return 0; // To indicate the end
  // Produce pointer to desired element:
  return storage[index];
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
