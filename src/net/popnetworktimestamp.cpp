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
#include "net/popnetworktimestamp.hpp"
#include "json/json.h"
#include "core/utilities.hpp"





namespace pop
{

	boost::asio::io_service popnetwork_timestamp_io_service;




        
        

	template <typename DATA_TYPE>
	void PopNetworkTimestamp<DATA_TYPE>::thread_run()
	{
		// this is blocking
		PopNetworkTimestamp::io_service.run();
	}

	template <typename DATA_TYPE>
	void PopNetworkTimestamp<DATA_TYPE>::start_receive()
	{
//            cout << "waiting for next packet" << endl;
//		socket_.async_receive_from(
//			boost::asio::buffer(recv_buffer_, POPNETWORK_SOCKET_MAXLEN), incoming_endpoint_,
//			boost::bind(&PopNetworkTimestamp::handle_receive, this,
//				boost::asio::placeholders::error,
//				boost::asio::placeholders::bytes_transferred));
	}

//        void PopNetworkTimestamp::test()
//        {
//
//        	PopRadio r;
//        	r.setBatCurrent(1);
//        	r.setBatVoltage(1);
//        	r.setLat(37.0);
//        	r.setLon(-122);
//        	r.setStatus(0);
//
//        	cout << r.seralize() << endl;
//
//
//
//
//        	return;
//            cout << "Hello world" << endl;
//
//
//            // construct programatically
//        json::object obj1;
//        obj1.insert("test1", "hello world")
//                .insert("test2", 10)
//                .insert("test3", json::object().insert("x", 123.456))
//                .insert("test4", json::array().append(1).append(2).append(3).append(4));
////
//        std::cout << json::pretty_print(obj1) << std::endl;
////
//        }
//
//        size_t PopNetworkTimestamp::get_length(uint8_t* data, size_t max_length)
//        {
//            if( max_length < 1 )
//                return 0;
//
//            if( data[0] < 126 )
//            {
//                return data[0];
//            }
//            else if( data[0] == 126 )
//            {
//                if( max_length < 3 )
//                    return 0;
//
//                size_t len;
//                len = (data[1] << 8) | data[2];
//                return len;
//            }
//            else if( data[0] == 127 )
//            {
//                return 0;
//                //FIXME: not implemented
//            }
//
//            return 0;
//        }
//
//        // returns how many bytes the length added to the message
//        // returns 0 in case of error
//        size_t PopNetworkTimestamp::get_length_from_header(uint8_t* data, size_t max_length)
//        {
//            if( max_length < 1 )
//               return 0;
//
//            if( data[0] < 126 )
//            {
//                return 1;
//            }
//            else if( data[0] == 126 )
//            {
//                if( max_length < 3 )
//                    return 0;
//                return 3;
//            }
//            else if( data[0] == 127 )
//            {
//                if( max_length < 9)
//                    return 0;
//                return 9;
//            }
//
//            return 0;
//        }
//
//    // only supports messages up to 65536 characters long
//    // takes a first param of byte_count and generates the correct "websocket style" header
//    // *header_len is set to the number of bytes used in header[]
//    void PopNetworkTimestamp::build_message_header(const size_t byte_count, char header[3], size_t* header_len)
//    {
//    	size_t len = byte_count;
//    	if( len < 126 )
//    	{
//    		header[0] = len & 0xff;
//    		*header_len = 1;
//    	}
//    	else if( len < 65536 )
//    	{
//    		header[0] = 126;
//    		header[1] = (len>>8) & 0xff;
//    		header[2] = len & 0xff;
//    		*header_len = 3;
//    	}
//    }
//
//    void PopNetworkTimestamp::setup_radio(PopRadio *r)
//    {
//    	r->setBatCurrent(1.0010101);
//    	r->setBatVoltage(1.12345);
//    	r->setLat(37);
//    	r->setLon(-122);
//    	r->setTemp(98);
//    	r->setStatus(0);
//    	r->setSerial(0);
//
//    }
//
//    void PopNetworkTimestamp::debug_pipe()
//    {
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
//    		PopNetworkTimestamp::setup_radio(&array[i]);
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
//
//    }
//
//	template <typename DATA_TYPE>
//	void PopNetworkTimestamp<DATA_TYPE>::process(const DATA_TYPE* data, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
//	{
//            cout << "called process" << endl;
//
////            debug_pipe();
////
////            static int debug = 0;
////
////
////            PopRadio r;
////            r.setBatCurrent(1.0010101);
////            r.setBatVoltage(1.12345);
////            r.setLat((debug++));
////            r.setLon(-122);
////            r.setTemp(98);
////            r.setStatus(0);
////            r.setSerial(0);
////
////
////
////            std::string str = r.seralize();
////            unsigned int json_len = str.size();
////
////            const char* json_c_str = str.c_str();
////
////            char message[1024];
////            unsigned int message_bytes = 0;
////
////            char header[3];
////            size_t header_len;
////            build_message_header(json_len, header, &header_len);
////
////
////            for(size_t i = 0; i < header_len; i++)
////                       {
////                           printf("%0x\r\n", header[i]);
////                       }
////
////
////            memcpy(message, header, header_len);
////            message_bytes += header_len;
////
////
////            memcpy(message+message_bytes, json_c_str, std::min((unsigned)1024,json_len) );
////            message_bytes += std::min((unsigned)1024,json_len);
////
////            printf("build header with length %ld for %d bytes\r\n", header_len, json_len);
////
////
////
////
//////            		socket_.send_to(boost::asio::buffer(data, NETWORK_PACKET_SIZE * sizeof(float)),outgoing_endpoint_);
////      		socket_.send_to(boost::asio::buffer(message, message_bytes),outgoing_endpoint_);
//
//
//	}

	template <typename DATA_TYPE>
	void PopNetworkTimestamp<DATA_TYPE>::handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred)
	{
            
//            cout << "received UDP packet" << endl;
//
////            float* buffer = PopSource<float>::get_buffer(bytes_transferred/sizeof(float));
//
//
//            // print raw bytes
//
//
//
//            size_t message_length = this->get_length((uint8_t*)recv_buffer_, bytes_transferred);
//            size_t header_length = this->get_length_from_header((uint8_t*)recv_buffer_, bytes_transferred);
//
//            if( bytes_transferred < message_length + header_length )
//            {
//                printf("Invalid or truncated message\r\n");
//            }
//            else
//            {
//                handle_json((uint8_t*)recv_buffer_+header_length, message_length);
//            }
//
//#ifdef DEBUG_POPNETWORK_JSON
//            for(int i = 0; i < bytes_transferred; i++)
//            {
//                printf("%0x\r\n", recv_buffer_[i]);
//            }
//            printf("Got length of: %ld with %ld extra bytes for header\r\n", message_length, header_length);
//            printf("Got %ld bytes\r\n", bytes_transferred);
//#endif
//
//            PopSource<float>::process();
//
//            // get next packet
//            start_receive();
            
	}
        
} // namespace pop
