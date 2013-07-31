/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <fstream>

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

class PopRadio : public PopSource<>
{

};
}
