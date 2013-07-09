/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#ifndef __POP_BLOCK_HPP_
#define __POP_BLOCK_HPP_

#include <popsink.hpp>
#include <popsource.hpp>

namespace pop
{


/**
 * Standard data processing block.
 */
template <class IN_TYPE = std::complex<float>,
         class OUT_TYPE = std::complex<float> >
class PopBlock :
    public PopSink<IN_TYPE>,
    public PopSource<OUT_TYPE>
{
public:
    /**
     * Class constructor.
     * @param reqBufSize Size of input buffer in number of samples. A value of
     * zero indicates that the class can accept any number of input samples.
     * @param sizeBuf Size of output buffer in number of samples.
     */
    PopBlock(size_t reqBufSize = 0, size_t sizeBuf = 0) :
        PopSink<IN_TYPE>(reqBufSize),
        PopSource<OUT_TYPE>(sizeBuf)
    {
    }

    /**
     * Class deconstructor.
     */
    ~PopBlock()
    {
    }


    virtual void process(IN_TYPE* in, OUT_TYPE* out, size_t size) = 0;


    void process(IN_TYPE* in, size_t size)
    {
        OUT_TYPE* buf;

        buf = PopSource<OUT_TYPE>::get_buffer();

        // crunch data per user application
        process(in, buf, size);

        // call attached sinks
        PopSource<OUT_TYPE>::process();
    }



};

} // namespace pop

#endif // __POP_BLOCK_HPP_
