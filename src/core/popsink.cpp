#include <core/popsink.hpp>


using namespace std;

namespace pop
{


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
#define TEMPLATE_TYPE_CNT 2


#define BOOST_PP_DEF(z, I, _) \
\
\
template void PopSink<TEMPLATE_TYPE(I)>::unblock(const TEMPLATE_TYPE(I)* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size );

BOOST_PP_REPEAT(TEMPLATE_TYPE_CNT, BOOST_PP_DEF, _)

#undef BOOST_PP_DEF





//template void PopSink<char>::unblock(const char* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size );
//template void PopSink<float>::unblock(const float* in, size_t size, const PopTimestamp* timestamp_in, size_t timestamp_size );


}
