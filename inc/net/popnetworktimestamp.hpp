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
		void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred);
		void start_receive();
//		void init();
		void thread_run();
		void test();
		size_t get_length(uint8_t* data, size_t max_length);
		size_t get_length_from_header(uint8_t* data, size_t max_length);
		void build_message_header(const size_t byte_count, char header[3], size_t* header_len);
		void handle_json(uint8_t* bytes, size_t len);
		static void setup_radio(PopRadio *r);
		void debug_pipe();


		boost::asio::ip::udp::socket socket_;
		int incoming_port_;
		int outgoing_port_;

		uint8_t *mp_buf; ///< circular buffer
		size_t m_buf_read_idx; ///< read index in samples
		size_t m_buf_write_idx; ///< write index in samples
		size_t m_buf_size; ///< circular buffer size in samples

		boost::asio::ip::udp::endpoint incoming_endpoint_;
		boost::asio::ip::udp::endpoint outgoing_endpoint_;

		char recv_buffer_[POPNETWORK_SOCKET_MAXLEN];


                
                boost::thread *m_pThread; ///< thead for boost's io_service.run();
                
                


	public:
			//PopNetworkTimestamp(int incoming_port = 5004, int outgoing_port = 35005, char* outgoing_ip = NULL);
//
//                template <typename DATA_TYPE>
//                	boost::asio::io_service PopNetworkTimestamp<DATA_TYPE>::io_service;

			PopNetworkTimestamp(int incoming_port, std::string outgoing_ip, int outgoing_port, int chunk = 0)
				: PopSink<DATA_TYPE>("PopNetworkTimestamp", chunk), PopSource<DATA_TYPE>("PopNetworkTimestamp"),
				  socket_(popnetwork_timestamp_io_service, udp::endpoint(udp::v4(), incoming_port)),
				incoming_port_(incoming_port), outgoing_port_(outgoing_port), mp_buf(0),
				m_buf_read_idx(0), m_buf_write_idx(0),
				m_buf_size(NETWORK_BUFFER_SIZE_BYTES / sizeof(NETWORK_STREAM_DATA_TYPE))
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
						outgoing_endpoint_.address(ip::address::from_string(outgoing_ip));
						outgoing_endpoint_.port(outgoing_port);

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
				if( this->m_rgSource )
				{
					// open an outgoing connection

					if( NETWORK_BUFFER_SIZE_BYTES % sizeof(NETWORK_STREAM_DATA_TYPE) )
							throw runtime_error("[POPNETWORK] - network stream datatype not divisible by packet length\r\n");


					cout << this->m_rgSource->get_name() << " is connected to us " << endl;
				}
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

	};
} // namespace pop


//template <typename DATA_TYPE>
//boost::asio::io_service PopNetworkTimestamp<DATA_TYPE>::io_service;


//template<>
//boost::asio::io_service PopNetworkComplex<PopSymbol>::io_service;




//boost::asio::io_service PopNetworkComplex::io_service;

#undef NETWORK_STREAM_DATA_TYPE

#endif
