
#ifndef __POPUTILS_H_
#define __POPUTILS_H_

#include <core/popsink.hpp>
#include <core/popsource.hpp>

#include "poptypes.h"

namespace pop
{

template <typename FORMATIN, typename FORMATOUT>
class PopTypeConversion : public PopSink<FORMATIN>, public PopSource<FORMATOUT>
{
public:
    PopTypeConversion() : PopSink<FORMATIN>("PopTypeConversion", 100), PopSource<FORMATOUT>("PopTypeConversion")
    {

    }
    void init() {}
    void process(const FORMATIN* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size)
    {
        size_t n;
        FORMATOUT* out;
        PopTimestamp* out_ts;

        out = PopSource<FORMATOUT>::get_buffer(size);

        out_ts = PopSource<FORMATOUT>::get_timestamp_buffer(timestamp_size);

        for( n = 0; n < size; n++)
            out[n] = data[n];

        for( n = 0; n < timestamp_size; n++)
            out_ts[n] = timestamp_data[n];

        PopSource<FORMATOUT>::process();
    }
};

}


#endif // __POPUTILS_H_
