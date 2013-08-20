/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <fstream>
#include <iostream>

#include <boost/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/sinc.hpp>

using namespace boost::posix_time;

#include "core/popblock.hpp"

namespace pop
{

struct odd
{
    uint8_t a, b, c;
};

class PopOdd : public PopSource<struct odd>
{
public:
    PopOdd() : PopSource<struct odd>("PopOdd") { }

    void start()
    {
        odd one[86];

        process(one, 86);
    }
};

class PopPoop : public PopSink<struct odd>
{
public:
    PopPoop() : PopSink<struct odd>("PopPoop") { }
    void process(const struct odd* data, size_t size)
    {
        printf("received %lu struct odd from PopPoop\r\n", size);
    }
    void init()
    {
        
    }
};

struct PopMsg
{
    char origin[20];
    char desc[20];
    uint16_t len;
    uint8_t data[0];
};

class PopBob : public PopSink<PopMsg>
{
public:
    PopBob() : PopSink<PopMsg>("PopBob") { }
    void init() { }
    void process(const PopMsg* data, size_t size)
    {
        printf("received %lu PopBob(s)\r\n", size);
    }

};

template <typename FORMAT>
class PopDecimate : public PopSink<FORMAT>, public PopSource<FORMAT>
{
public:
    PopDecimate(unsigned rate = 2) : PopSink<FORMAT>("PopDecimate"), PopSource<FORMAT>("PopDecimate"), m_rate(rate)
    {

    }
    void init() {}
    void process(const FORMAT* data, size_t size)
    {
        size_t n;
        FORMAT* out;

        out = PopSource<FORMAT>::get_buffer(size/m_rate);

        for( n = 0; n < size/m_rate; n++)
            out[n] = data[n*m_rate];

        PopSource<FORMAT>::process();
    }
    unsigned m_rate;
};


class PopAlice : public PopSource<PopMsg>
{
public:
    PopAlice() : PopSource<PopMsg>("PopAlice") { }

    void send_message(const char* desc, void*, size_t bytes)
    {

    }
    void start()
    {
        PopMsg *msg = (PopMsg*)malloc(sizeof(PopMsg) + 10);

        process(msg, 1);
    }
};

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

        while(1)
        {
            a = get_buffer(12);
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
    PopIntAdd() : PopBlock("PopIntAdd", 100) { }
    void init() { }
    void process(const uint8_t* in, uint8_t* out, size_t size)
    {
        for( unsigned n = 0; n < size; n++ )
        {
            out[n] = in[n] + 0x20;
        }
    }
};

template <typename T>
class PopDumpToFile : public PopSink<T>
{
public:
    PopDumpToFile(const char* file_name = "dump.raw") : PopSink<T>("PopDumpToFile"),
        m_fileName(file_name)
    {
        printf("%s - created %s file\r\n", PopSink<T>::get_name(), m_fileName);
        m_fs.open(m_fileName, std::ofstream::binary);
    }
    ~PopDumpToFile()
    {
        m_fs.close();
    }
private:
    void init()
    {
    }
    void process(const T* in, size_t size)
    {
        printf("+");
        size_t bytes = size * sizeof(T);
        m_fs.write((const char*)in, bytes);
    }
    std::ofstream m_fs;
    const char* m_fileName;
};

class PopMagnitude : public PopSink<std::complex<float> >, public PopSource<float>
{
public:
    PopMagnitude() : PopSink<std::complex<float> >("PopMagnitude", 65536), PopSource<float>("PopMagnitude") { }
private:
    void init() { }
    void process(const std::complex<float>* in, size_t size)
    {
        float *out = get_buffer(size);

        for( size_t n = 0; n < size; n++ )
            out[n] = std::abs(in[n]);

        PopSource<float>::process();
    }
};

class PopWeightSideBand : public PopSink<std::complex<float> >, public PopSource<uint8_t>
{
public:
    PopWeightSideBand() :PopSink<std::complex<float> >("PopWeightSideBand", 1040), PopSource<uint8_t>("PopWeightSideBand") { }
private:
    void init() { }
    void process(const std::complex<float>* in, size_t size)
    {
        size_t n;
        std::complex<float> theta1, theta2;
        uint8_t *out;

        out = get_buffer(size / 8);

        memset(out, 0, size / 8);

        theta1 = in[-1] / std::abs(in[-1]);
        for( n = 0; n < size; n++ )
        {
            theta2 = in[n] / std::abs(in[n]);

            // the magic sauce, dawg!!!
            out[n/8] |= ((~((uint32_t)(fmod(std::arg(theta1) + M_PI + M_PI - std::arg(theta2), M_PI*2) - M_PI))>>31))<<(n%8);

            theta1 = theta2;

            /*if( n%8 == 7 )
                printf("%02hhx", out[n/8]);*/
        }
        //printf("\r\n");

        PopSource<uint8_t>::process();
    }
};

#define SINC_SAMPLES_BACK 10
#define SINC_SAMPLES_FORWARD 10
#define SINC_SAMPLE_PERIOD 1.96923e-5

class PopGmskDemod : public PopSink<std::complex<float> >, public PopSource<std::complex<float> >
{
public:
    PopGmskDemod() : PopSink<std::complex<float> >("PopGmskDemod", 1040), PopSource<std::complex<float> >("PopGmskDemod") { }
    ~PopGmskDemod() { }
private:
    void init() { }
    void process(const std::complex<float>* in, size_t size)
    {
        signed n, m, p, idx, idx2;
        std::complex<float>* buf;

        // TODO: get timestamp from attached PopSource

        //buf = (std::complex<float>*)malloc( 1040 * 3 * sizeof(std::complex<float>) );

        buf = get_buffer(1040);

        // perform a sinc interpolation
        /*for( n = 0; n < (signed)size; n++ )
        {
            // scan across arrival phase
            for( m = 0; m < 3; m++ )
            {
                idx = n * 3 + m;
                buf[idx].real(0.0);
                buf[idx].imag(0.0);

                // integrate
                for( p = -SINC_SAMPLES_BACK; p < SINC_SAMPLES_FORWARD - 1; p++ )
                {
                    idx2 = n + p - SINC_SAMPLES_FORWARD;
                    buf[idx] += in[idx2] * boost::math::sinc_pi<float>( (idx - idx2 * SINC_SAMPLE_PERIOD) / SINC_SAMPLE_PERIOD );
                }
            }
        }*/

        PopSource<std::complex<float> >::process();
    }
};

class PopWeightSideBandDebug : public PopSink<std::complex<float> >, public PopSource<std::complex<float> >
{
public:
    PopWeightSideBandDebug() :PopSink<std::complex<float> >("PopWeightSideBandDebug", 1040), PopSource<std::complex<float> >("PopWeightSideBandDebug") { }
private:
    void init() { }
    void process(const std::complex<float>* in, size_t size)
    {
        size_t n;
        std::complex<float> theta1, theta2;
        std::complex<float> *out;
        float a;

        out = get_buffer(size);

        theta1 = in[-1] / std::abs(in[-1]);
        for( n = 0; n < size; n++ )
        {
            theta2 = in[n] / std::abs(in[n]);
            a = (fmod(std::arg(theta1) + M_PI + M_PI - std::arg(theta2), M_PI*2) - M_PI)*10.0;
            out[n].real(a);
            if( a > 0.0 )
                out[n].real(30);
            else
                out[n].real(-30);
            out[n].imag(std::abs(in[n]));

            theta1 = theta2;

        }

        PopSource<std::complex<float> >::process();
    }
};

const char pn_code_a[] = {0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
                          0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
                          0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
                          0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
                          0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00};

const uint8_t pn_code_b[] = {
       0x67,0x7A,0xFA,0x1C,0x52,0x07,0x56,0x06,0x08,0x5C,0xBF,0xE4,0xE8,0xAE,0x88,0xDD,
       0x87,0xAA,0xAF,0x9B,0x04,0xCF,0x9A,0xA7,0xE1,0x94,0x8C,0x25,0xC0,0x2F,0xB8,0xA8,
       0xC0,0x1C,0x36,0xAE,0x4D,0x6E,0xBE,0x1F,0x99,0x0D,0x4F,0x86,0x9A,0x65,0xCD,0xEA,
       0x03,0xF0,0x92,0x52,0xDC,0x20,0x8E,0x69,0xFB,0x74,0xE6,0x13,0x2C,0xE7,0x7E,0x25,
       0xB5,0x78,0xFD,0xFE,0x33,0xAC,0x37,0x2E,0x6B,0x83,0xAC,0xB0,0x22,0x00,0x23,0x97,
       0xA6,0xEC,0x6F,0xB5,0xBF,0xFC,0xFD,0x4D,0xD4,0xCB,0xF5,0xED,0x1F,0x43,0xFE,0x58,
       0x23,0xEF,0x4E,0x82,0x32,0xD1,0x52,0xAF,0x0E,0x71,0x8C,0x97,0x05,0x9B,0xD9,0x82};

class PopDigitalDeconvolve : public PopSink<uint8_t>, public PopSource<std::complex<float> >
{
public:
    PopDigitalDeconvolve() : PopSink<uint8_t>("PopDigitalDeconvolve", 1040 / 8), PopSource<std::complex<float> >("PopDigitalDeconvolve") { }
private:
    void init() { }
    void process(const uint8_t * in, size_t size)
    {
        signed n, m;
        signed B, b;
        unsigned s;
        uint8_t j, k;
        ptime t1;
        t1 = microsec_clock::local_time();
        std::complex<float>* buf;

        buf = get_buffer(size*8);

        for( n = 0; n < size; n++ )
        {
            s = 0;
            for( m = 0; m < 400; m++ )
            {
                B = m / 3 / 8;
                b = (m / 3) % 8;
                j = (in[n - 50 + B] >> b) & 0x01;
                k = (pn_code_b[B] >> (7-b)) & 0x01;
                s += !(j ^ k); // XNOR
            }
            if( s > 300 )
            {
                std::cout << to_iso_extended_string(t1) << " - ";
                printf("signal: %u\r\n", s);
            }
            buf[n].real(s);
            buf[n].imag(s);
        }

        PopSource<std::complex<float> >::process();
    }
};

class PopRadio : public PopSource<>
{

};
}