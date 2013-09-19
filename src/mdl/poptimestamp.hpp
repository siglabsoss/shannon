#ifndef __POP_TIMESTAMP_HPP_
#define __POP_TIMESTAMP_HPP_

#include <time.h>

namespace pop
{

class PopTimestamp : public timespec
{
public:

	unsigned int offset;


};



} // namespace pop



#endif
