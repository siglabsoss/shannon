#ifndef __POP_WEBHOOK_HPP_
#define __POP_WEBHOOK_HPP_

#include "core/popsink.hpp"
#include "popradio.h"

namespace pop
{

class PopWebhook : public PopSink<PopRadio>
{
public:
	PopWebhook(unsigned notused);
	void process(const PopRadio* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size);
	void init() {}


private:
	void doHook(std::string url, std::string data);


};

}

#endif
