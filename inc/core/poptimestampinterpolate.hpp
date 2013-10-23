#ifndef __POP_TIME_INTERPOLATE_HPP_
#define __POP_TIME_INTERPOLATE_HPP_



#include <core/popsink.hpp>
#include <core/popsource.hpp>

using namespace std;

namespace pop
{

template <typename DATA_TYPE>
class PopTimestampInterpolation : public PopSink<DATA_TYPE>, public PopSource<DATA_TYPE>
{
private:
	// pointer to a single timestamp given to us in the previous call to process
	const PopTimestamp* previous_timestamp;

	// the number of data samples from the previous call to process
	size_t previous_size;
public:
	PopTimestampInterpolation(size_t chunk) : PopSink<DATA_TYPE>("PopTimestampInterpolation", chunk), PopSource<DATA_TYPE>("PopTimestampInterpolation"), previous_timestamp(0), previous_size(0)
    {

    }
    void init() {}
    void process(const DATA_TYPE* data, size_t size, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
    {
    	if( timestamp_size != 1 )
    	{
    		cout << this->m_rgSource->get_name() << " sent " << timestamp_size << " timestamps which is different than the expected count of " << 1 << endl;
    	}

    	// handle initial conditions
    	if( previous_timestamp == 0 )
    	{
    		previous_timestamp = timestamp_data;
    		previous_size = size;
    		return;
    	}


    	// pointer arithmetic to negatively index into buffer
    	const DATA_TYPE* previous_data = data - previous_size;

    	DATA_TYPE* out;
    	PopTimestamp* out_ts;

    	out_ts = PopSource<DATA_TYPE>::get_timestamp_buffer(previous_size);

    	size_t n;

		// create mutable copy
		PopTimestamp time_difference = PopTimestamp(timestamp_data[0]);

		// calculate difference using -= overload (which should be most accurate)
		time_difference -= *previous_timestamp;

		// calculate time per sample
		double time_per_sample = time_difference.get_real_secs() / previous_size;

//		cout << "    with time per sample of " << boost::lexical_cast<string>(time_per_sample) << endl;

		// The exact timestamp of the sample
		PopTimestamp exact_timestamp;

    	for( n = 0; n < previous_size; n++)
    	{
    		// copy constructor
    		exact_timestamp = PopTimestamp(*previous_timestamp);

    		// interpolate for the nth sample
    		exact_timestamp += time_per_sample * n;

    		// save to buffer
    		out_ts[n] = PopTimestamp(exact_timestamp);
    	}

    	// above, we omit calling get_buffer() and let PopSource copy the data samples for us
    	PopSource<DATA_TYPE>::process(previous_data, previous_size, out_ts, previous_size);

    	// save pointers
		previous_timestamp = timestamp_data;
		previous_size = size;
    }
};

}


#endif
