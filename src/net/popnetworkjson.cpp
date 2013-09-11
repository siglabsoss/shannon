/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include "net/popnetworkjson.hpp"
#include "json/json.h"
#include "dsp/utils.hpp"


using boost::asio::ip::udp;
using namespace boost::asio;
using namespace std;

#define NETWORK_PACKET_SIZE 368 // in samples
#define NETWORK_STREAM_DATA_TYPE char
#define NETWORK_BUFFER_SIZE_BYTES (NETWORK_PACKET_SIZE * 100 * sizeof(NETWORK_STREAM_DATA_TYPE))   // in bytes

#define OUTGOING_IP_ADDRESS "192.168.1.41"
//#define OUTGOING_IP_ADDRESS "173.167.119.220"

namespace pop
{

	boost::asio::io_service PopNetworkJson::io_service;

	PopNetworkJson::PopNetworkJson(int incoming_port, int outgoing_port)
	: PopSink<float>("PopNetworkJson", 4), PopSource<float>("PopNetworkJson"), socket_(io_service, udp::endpoint(udp::v4(), incoming_port)),
	incoming_port_(incoming_port), outgoing_port_(outgoing_port), mp_buf(0),
	m_buf_read_idx(0), m_buf_write_idx(0),
	m_buf_size(NETWORK_BUFFER_SIZE_BYTES / sizeof(NETWORK_STREAM_DATA_TYPE)), m_pThread(0), m_packets_received(0)
	{
//		recv_buffer_.resize(12000);

            
             start_receive();
             
             // start network thread to respoind to incomming packets
             if( 0 == m_pThread )
                m_pThread = new boost::thread(boost::bind(&PopNetworkJson::thread_run, this));
            
             


		cout << "initialized outgoing UDP socket" << endl;

		/* We set the outgoing address on the first incoming packet
		   and set the outgoing port here. */
		outgoing_endpoint_.address(ip::address::from_string(OUTGOING_IP_ADDRESS));
		outgoing_endpoint_.port(outgoing_port);

		if( NETWORK_BUFFER_SIZE_BYTES % sizeof(NETWORK_STREAM_DATA_TYPE) )
			throw runtime_error("[POPNETWORK] - network stream datatype not divisible by packet length\r\n");

		mp_buf = (uint8_t*)malloc(NETWORK_BUFFER_SIZE_BYTES);
                

	}

	PopNetworkJson::~PopNetworkJson()
	{
		if( mp_buf ) free(mp_buf);
	}

	void PopNetworkJson::init()
	{
	}
        
        
        
        void PopNetworkJson::thread_run()
        {
            // this is blocking
            PopNetworkJson::io_service.run();
        }

	void PopNetworkJson::start_receive()
	{
//            cout << "waiting for next packet" << endl;
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_, POPNETWORK_SOCKET_MAXLEN), incoming_endpoint_,
			boost::bind(&PopNetworkJson::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

        void PopNetworkJson::test()
        {

        	PopRadio r;
        	r.setBatCurrent(1);
        	r.setBatVoltage(1);
        	r.setLat(37.0);
        	r.setLon(-122);
        	r.setStatus(0);

        	cout << r.seralize() << endl;




        	return;
            cout << "Hello world" << endl;
            
            
            // construct programatically
        json::object obj1;
        obj1.insert("test1", "hello world")
                .insert("test2", 10)
                .insert("test3", json::object().insert("x", 123.456))
                .insert("test4", json::array().append(1).append(2).append(3).append(4));
//
        std::cout << json::pretty_print(obj1) << std::endl;
//        
        }
        
        size_t PopNetworkJson::get_length(uint8_t* data, size_t max_length)
        {
            if( max_length < 1 )
                return 0;
            
            if( data[0] < 126 )
            {
                return data[0];
            }
            else if( data[0] == 126 )
            {
                if( max_length < 3 )
                    return 0;
                
                size_t len;
                len = (data[1] << 8) | data[2];
                return len;
            } 
            else if( data[0] == 127 )
            {
                return 0;
                //FIXME: not implemented
            }

            return 0;
        }
        
        // returns how many bytes the length added to the message
        // returns 0 in case of error
        size_t PopNetworkJson::get_length_from_header(uint8_t* data, size_t max_length)
        {
            if( max_length < 1 )
               return 0;
            
            if( data[0] < 126 )
            {
                return 1;
            }
            else if( data[0] == 126 )
            {
                if( max_length < 3 )
                    return 0;
                return 3;
            } 
            else if( data[0] == 127 )
            {
                if( max_length < 9)
                    return 0;
                return 9;
            }

            return 0;
        }



    void PopNetworkJson::setup_radio(PopRadio *r)
    {
    	r->setBatCurrent(1.0010101);
    	r->setBatVoltage(1.12345);
    	r->setLat(37);
    	r->setLon(-122);
    	r->setTemp(98);
    	r->setStatus(0);
    	r->setSerial(0);

    }

    void PopNetworkJson::debug_pipe()
    {
//    	int number_radios = 5;
//    	int i;
//
////    	setup random
//    	srand(time(0));
//
//    	PopRadio* array = new PopRadio[number_radios];
//
//    	for(i = 0; i < number_radios; i++)
//    	{
//    		PopNetworkJson::setup_radio(&array[i]);
//
//    		array[i].setLat(RAND_BETWEEN(37.0,38.0));
//    		array[i].setLon(RAND_BETWEEN(-122.4,-123));
//    		array[i].setSerial(i);
//
//
////    		printf("lat: %f\n", array[i].getLon());
//    	}
//
//    	char message[1024];
//
////    	send over wire
//    	for(i = 0; i < number_radios; i++)
//    	{
//
//    		std::string str = array[i].seralize();
//    		unsigned int json_len = str.size();
//    		const char* json_c_str = str.c_str();
//
//    		char message[1024];
//    		unsigned int message_bytes = 0;
//
//    		char header[3];
//    		size_t header_len;
//    		build_message_header(json_len, header, &header_len);
//
//
//    		memcpy(message, header, header_len);
//    		message_bytes += header_len;
//
//
//    		memcpy(message+message_bytes, json_c_str, std::min((unsigned)1024,json_len) );
//    		message_bytes += std::min((unsigned)1024,json_len);
//
////    		printf("build header with length %ld for %d bytes\r\n", header_len, json_len);
//
//    		socket_.send_to(boost::asio::buffer(message, message_bytes),outgoing_endpoint_);
//    	}
//
//
//
//    	delete[] array;

    }
        
	void PopNetworkJson::process(const float* data, size_t len)
	{
            cout << "called process" << endl;

            debug_pipe();

            static int debug = 0;


            PopRadio r;
            r.setBatCurrent(1.0010101);
            r.setBatVoltage(1.12345);
            r.setLat((debug++));
            r.setLon(-122);
            r.setTemp(98);
            r.setStatus(0);
            r.setSerial(0);



            std::string str = r.seralize();
            unsigned int json_len = str.size();

            const char* json_c_str = str.c_str();

            char message[1024];
            unsigned int message_bytes = 0;

            char header[3];
            size_t header_len;
            PopSource::build_message_header(json_len, header, &header_len);


            for(size_t i = 0; i < header_len; i++)
                       {
                           printf("%0x\r\n", header[i]);
                       }


            memcpy(message, header, header_len);
            message_bytes += header_len;


            memcpy(message+message_bytes, json_c_str, std::min((unsigned)1024,json_len) );
            message_bytes += std::min((unsigned)1024,json_len);

            printf("build header with length %ld for %d bytes\r\n", header_len, json_len);



            if( m_packets_received != 0 )
            	socket_.send_to(boost::asio::buffer(message, message_bytes),incoming_endpoint_);

                
	}

	void PopNetworkJson::handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred)
	{
            
            cout << "received UDP packet number " << m_packets_received << endl;
            
            m_packets_received++;

//            float* buffer = PopSource<float>::get_buffer(bytes_transferred/sizeof(float));

               
            // print raw bytes
            
           
            
            size_t message_length = this->get_length((uint8_t*)recv_buffer_, bytes_transferred);
            size_t header_length = this->get_length_from_header((uint8_t*)recv_buffer_, bytes_transferred);
            
            if( bytes_transferred < message_length + header_length )
            {
                printf("Invalid or truncated message\r\n");
            }
            else
            {
                handle_json((uint8_t*)recv_buffer_+header_length, message_length);
            }
            
#ifdef DEBUG_POPNETWORK_JSON
            for(int i = 0; i < bytes_transferred; i++)
            {
                printf("%0x\r\n", recv_buffer_[i]);
            }
            printf("Got length of: %ld with %ld extra bytes for header\r\n", message_length, header_length);
            printf("Got %ld bytes\r\n", bytes_transferred);
#endif
            
            PopSource<float>::process();

            // get next packet
            start_receive();
            
	}
        
        void PopNetworkJson::handle_json(uint8_t* bytes, size_t len)
        {
            std::string s( reinterpret_cast<char const*>(bytes), len );
            
            cout << "got this string: " << s << endl;
            
            json::value json = json::parse(s);
            
            const json::value jlat = json["lat"];
            double lat = json::to_number(jlat);
            double lon = json::to_number(json["lon"]);
            string id = json::to_string(json["id"]);
            
            cout << "lat: " << lat << endl;
            cout << "lon: " << lon << endl;
            cout << "id: "  << id << endl;
        }
} // namespace pop
