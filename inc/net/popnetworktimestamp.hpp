#ifndef __POP_NETWORK_TIMESTAMP__
#define __POP_NETWORK_TIMESTAMP__

/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"
#include "popradio.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include "net/popnetworktimestamp.hpp"
#include "json/json.h"
#include "core/utilities.hpp"


// FIXME
#undef NETWORK_STREAM_DATA_TYPE
#undef NETWORK_BUFFER_SIZE_BYTES


#define POPNETWORK_SOCKET_MAXLEN (1024)


#define NETWORK_PACKET_SIZE 368 // in samples
#define NETWORK_STREAM_DATA_TYPE float
#define NETWORK_BUFFER_SIZE_BYTES (NETWORK_PACKET_SIZE * 100 * sizeof(NETWORK_STREAM_DATA_TYPE))   // in bytes

#define NETWORK_UDP_PACKET_MTU (1500)

//#define OUTGOING_IP_ADDRESS "192.168.1.41"
//#define OUTGOING_IP_ADDRESS "173.167.119.220"



using boost::asio::ip::udp;
using namespace boost::asio;
using namespace std;



namespace pop
{

	extern boost::asio::io_service popnetwork_timestamp_io_service;


	template <typename DATA_TYPE>
	class PopNetworkTimestamp : public PopSink<DATA_TYPE>, public PopSource<DATA_TYPE>
	{
	public:







//		void process(const DATA_TYPE* data, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction);


	private:

//		template <typename DATA_TYPE>
//			boost::asio::io_service io_service;//


                // forward declare stuff
//		void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred);
		void start_receive();
//		void init();


		boost::asio::ip::udp::socket socket_;
		int incoming_port_;
		int outgoing_port_;
		std::string outgoing_ip_;

		uint8_t *mp_buf; ///< circular buffer
		size_t m_buf_read_idx; ///< read index in samples
		size_t m_buf_write_idx; ///< write index in samples
		size_t m_buf_size; ///< circular buffer size in samples

		boost::asio::ip::udp::endpoint incoming_endpoint_;
		boost::asio::ip::udp::endpoint outgoing_endpoint_;

		char recv_buffer_[NETWORK_UDP_PACKET_MTU];


                
                boost::thread *m_pThread; ///< thead for boost's io_service.run();
                
                


	public:
			//PopNetworkTimestamp(int incoming_port = 5004, int outgoing_port = 35005, char* outgoing_ip = NULL);
//
//                template <typename DATA_TYPE>
//                	boost::asio::io_service PopNetworkTimestamp<DATA_TYPE>::io_service;

			PopNetworkTimestamp(int incoming_port, std::string outgoing_ip, int outgoing_port, int chunk = 0)
				: PopSink<DATA_TYPE>("PopNetworkTimestamp", chunk), PopSource<DATA_TYPE>("PopNetworkTimestamp"),
				  socket_(popnetwork_timestamp_io_service, udp::endpoint(udp::v4(), incoming_port)),
				incoming_port_(incoming_port), outgoing_port_(outgoing_port), outgoing_ip_(outgoing_ip), mp_buf(0),
				m_buf_read_idx(0), m_buf_write_idx(0),
				m_buf_size(NETWORK_BUFFER_SIZE_BYTES / sizeof(NETWORK_STREAM_DATA_TYPE)),
				m_pThread(0)
				{

//
//
//				start_receive();
//
//				// start network thread to respoind to incomming packets
//				if( 0 == m_pThread )
//					m_pThread = new boost::thread(boost::bind(&PopNetworkJson::thread_run, this));
//



//						cout << "initialized outgoing UDP socket" << endl;

						/* We set the outgoing address on the first incoming packet
						   and set the outgoing port here. */


//						if( NETWORK_BUFFER_SIZE_BYTES % sizeof(NETWORK_STREAM_DATA_TYPE) )
//							throw runtime_error("[POPNETWORK] - network stream datatype not divisible by packet length\r\n");
//
//						mp_buf = (uint8_t*)malloc(NETWORK_BUFFER_SIZE_BYTES);


				}








			~PopNetworkTimestamp()
			{
		//		if( mp_buf ) free(mp_buf);
			}

			void wakeup()
			{
				cout << "awake" << endl;

				// if there is a source connected to us
				if( this->m_rgSource != NULL )
				{
					// We assume these were set correctly in the constructor
					outgoing_endpoint_.address(ip::address::from_string(outgoing_ip_));
					outgoing_endpoint_.port(outgoing_port_);

					// open an outgoing connection

					if( NETWORK_BUFFER_SIZE_BYTES % sizeof(NETWORK_STREAM_DATA_TYPE) )
							throw runtime_error("[POPNETWORK] - network stream datatype not divisible by packet length\r\n");


//					cout << this->m_rgSource->get_name() << " is connected to us for input" << endl;
				}

				// if there is a sink connected to us
				if( this->m_rgSinks.size() != 0 )
				{
					// open the udp port.
					// How is this different than the socket_ constructor in this class's constructor?
					incoming_endpoint_.port(incoming_port_);

					// prep for first packet
					receive_next();

					// start networking run loop in a new thread
					if( 0 == m_pThread )
						m_pThread = new boost::thread(boost::bind(&PopNetworkTimestamp::start_io_run_loop, this));

//					cout << "we connected to " << this->m_rgSinks[0]->get_name() << " for output " << endl;
				}
			}

			void start_io_run_loop()
			{
				// this is blocking
				popnetwork_timestamp_io_service.run();
			}


			void init()
			{

			}

			void process(const DATA_TYPE* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size, size_t timestamp_buffer_correction)
			{
				if( timestamp_size != 0 )
					cout << "This networking class drops all outgoing timestamps" << endl;

				// calculate how many bytes on the wire
				size_t outputByteCount = data_size * sizeof(DATA_TYPE);

				// throw warning (TODO this can worked around by sending multiple packets per process)
				if( outputByteCount > NETWORK_UDP_PACKET_MTU )
					cout << "You asked this networking class to transmit " << outputByteCount << " bytes, but it can't send more than " << NETWORK_UDP_PACKET_MTU << " at a time." << endl;

				// send packet on the wire
				socket_.send_to(boost::asio::buffer(data, outputByteCount), outgoing_endpoint_);

//				cout << "sending " << outputByteCount << " bytes." << endl;
			}


			void receive_next()
			{
//				cout << "waiting for next packet" << endl;
				socket_.async_receive_from(
						boost::asio::buffer(recv_buffer_, NETWORK_UDP_PACKET_MTU), incoming_endpoint_,
						boost::bind(&PopNetworkTimestamp::handle_receive, this,
								boost::asio::placeholders::error,
								boost::asio::placeholders::bytes_transferred));
			}

			void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred)
				{

			            cout << "received UDP packet" << endl;

//			//            float* buffer = PopSource<float>::get_buffer(bytes_transferred/sizeof(float));
//
//
//			            // print raw bytes
//
//
//
//			            size_t message_length = this->get_length((uint8_t*)recv_buffer_, bytes_transferred);
//			            size_t header_length = this->get_length_from_header((uint8_t*)recv_buffer_, bytes_transferred);
//
//			            if( bytes_transferred < message_length + header_length )
//			            {
//			                printf("Invalid or truncated message\r\n");
//			            }
//			            else
//			            {
//			                handle_json((uint8_t*)recv_buffer_+header_length, message_length);
//			            }
//
//			#ifdef DEBUG_POPNETWORK_JSON
//			            for(int i = 0; i < bytes_transferred; i++)
//			            {
//			                printf("%0x\r\n", recv_buffer_[i]);
//			            }
//			            printf("Got length of: %ld with %ld extra bytes for header\r\n", message_length, header_length);
//			            printf("Got %ld bytes\r\n", bytes_transferred);
//			#endif
//
////			            DATA_TYPE* dataOut = get_buffer(bytes_transferred);

			            size_t objectCount = bytes_transferred / sizeof(DATA_TYPE);

			            if( bytes_transferred % sizeof(DATA_TYPE) != 0 )
			            	cout << "received " << bytes_transferred << " bytes which is not evenly divisible by datatype size of " << sizeof(DATA_TYPE) << ".  Discarding the last partial object!" << endl;

			            // send to our sinks with no timestamps (cuz this class drops them)
			            PopSource<DATA_TYPE>::process((DATA_TYPE*) recv_buffer_, objectCount, NULL, 0);

			            // wait for next packet
			            receive_next();

				}

	};
} // namespace pop


#undef NETWORK_STREAM_DATA_TYPE

#endif
