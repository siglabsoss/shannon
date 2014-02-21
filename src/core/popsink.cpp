#include <core/popsink.hpp>

#include "mdl/poppeak.hpp"

using namespace std;

namespace pop
{



/**
 * Class constructor.
 * @param nInBuf Size of input buffer in number of samples. A value of
 * zero indicates that the class can accept any number of input samples.
 */
template <typename IN_TYPE>
PopSink<IN_TYPE>::PopSink(const char* name, size_t nInBuf) : PopObject(name), m_reqBufSize(nInBuf),
    m_sourceBufIdx(0), m_timestampSourceBufIdx(0), m_pThread(0), m_rgSource(0)
{
}

/**
 * Class destructor.
 */
template <typename IN_TYPE>
PopSink<IN_TYPE>::~PopSink()
{
    delete m_pThread;
}

/**
 * Start Thread
 */
template <typename IN_TYPE>
void PopSink<IN_TYPE>::start_thread()
{
	if( 0 == m_pThread )
	{
		m_mutex.lock();
		m_pThread = new boost::thread(boost::bind(&PopSink::run, this));
		m_mutex.lock();
	}
}

/**
 * Returns requested sample size for sink.
 */
template <typename IN_TYPE>
size_t PopSink<IN_TYPE>::sink_size() {
	return m_reqBufSize;
}

/**
 * Thread loop.
 */
template <typename IN_TYPE>
void PopSink<IN_TYPE>::run()
{
	buffer_read_pointer<IN_TYPE> buf;

	init();

	m_mutex.unlock();

	while(1)
	{
		this->wait_and_pop( buf );

		process( buf.data, buf.len, buf.timestamp_data, buf.timestamp_len );
	}
}


template <typename IN_TYPE>
void PopSink<IN_TYPE>::unblock(const IN_TYPE* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size )
{
    // check to for a valid amount of input samples
    if( 0 != m_reqBufSize )
        if( size != m_reqBufSize )
            throw PopException( msg_passing_invalid_amount_of_samples, get_name() );

    if( 0 == size )
        throw PopException( msg_passing_invalid_amount_of_samples, get_name() );

    if( m_pThread )
        this->push( buffer_read_pointer<IN_TYPE>(in, size, timestamp_in, timestamp_size ) );
    else
        process( in, size, timestamp_in, timestamp_size );
}



// This is a list of all the types that this templated class will use.
// Make sure to also update the CNT
// See this for information on wtf is happening: http://www.boost.org/doc/libs/1_55_0/libs/preprocessor/doc/topics/techniques.html
// See this for information about why: http://www.parashift.com/c++-faq-lite/templates-defn-vs-decl.html
#define TEMPLATE_TYPE(I) TEMPLATE_TYPE ## I
#define TEMPLATE_TYPE0    float
#define TEMPLATE_TYPE1    char
#define TEMPLATE_TYPE2	  pop::PopPeak
#define TEMPLATE_TYPE3    std::complex<float>
#define TEMPLATE_TYPE_CNT 4


#define BOOST_PP_DEF(z, I, _) \
\
\
template PopSink<TEMPLATE_TYPE(I)>::PopSink(const char* name, size_t nInBuf); \
template PopSink<TEMPLATE_TYPE(I)>::~PopSink(); \
template void PopSink<TEMPLATE_TYPE(I)>::start_thread(); \
template size_t PopSink<TEMPLATE_TYPE(I)>::sink_size(); \
template void PopSink<TEMPLATE_TYPE(I)>::run(); \
template void PopSink<TEMPLATE_TYPE(I)>::unblock(const TEMPLATE_TYPE(I)* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size );



BOOST_PP_REPEAT(TEMPLATE_TYPE_CNT, BOOST_PP_DEF, _)
#undef BOOST_PP_DEF

}
