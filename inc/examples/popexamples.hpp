/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include "core/popblock.hpp"

namespace pop
{


class PopPiSource : public PopSource<uint8_t>
{
public:
    PopPiSource() : PopSource<uint8_t>("PopPiSource"), b(0) { }
public:
    int b;
    void start()
    {
        unsigned n;
        uint8_t* a;
        resize_buffer(12);

        while(1)
        {
            a = get_buffer();
            for( n = 0; n < 12; n++ )
                a[n] = n;
            printf("%d\r\n", b);
            b += 12;
            process();
            boost::posix_time::milliseconds workTime(50);
            boost::this_thread::sleep(workTime);
        }
    }
};

class PopDummySink : public PopSink<>
{
public:
    void init() { }
    void process(const std::complex<float>* data, size_t size)
    {
        static int a = 0;

        a++;
        a %= 500;

        if( a == 0 )
        printf("received %lu samples (500 times)\r\n", size);
    }
};


class PopAdd: public PopBlock<>
{
    void init() { }
    void process(const std::complex<float>* in, std::complex<float>* out,
        size_t size)
    {
        size_t n;

        for( n = 0; n < size; n++)
            out[n] = in[n] + std::complex<float>(3.1415, 3.1415);
    }
};

class PopAddPi : public PopBlock<float, float>
{
    void init() { }
    void process(const float* in, float* out, size_t size)
    {
        size_t n;

        for( n = 0; n < size; n++)
            out[n] = in[n] + 3.141592;
    }
};



class PopTest1 : public PopSink<uint8_t>
{
public:
	PopTest1() : PopSink<uint8_t>("PopTest1", 11) {}
    void init() { }
    void process(const uint8_t* data, size_t size)
    {
        unsigned n;
        for(n=0;n<11;n++)
            printf("%02x ", data[n]);
        /*printf("... ");
        for(n=290;n<300;n++)
            printf("%02x ", data[n]);*/
        printf("\r\n");
    }
};

class PopIntAdd : public PopBlock<uint8_t, uint8_t>
{
public:
    PopIntAdd() : PopBlock("PopIntAdd", 100, 100) { }
    void init() { }
    void process(const uint8_t* in, uint8_t* out, size_t size)
    {
        for( unsigned n = 0; n < size; n++ )
        {
            out[n] = in[n] + 0x20;
        }
    }
};

class PopMagnitude : public PopBlock<std::complex<float>, float>
{
public:
    PopMagnitude() : PopBlock<std::complex<float>, float>("PopMagnitude", 65536,65536) { }
private:
    void init() { }
    void process(const std::complex<float>* in, float* out, size_t size)
    {
        for( size_t n = 0; n < size; n++ )
            out[n] = std::abs(in[n]);
    }
};

class PopRadio : public PopSource<>
{

};
}
