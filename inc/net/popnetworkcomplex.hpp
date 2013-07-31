/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include <complex>
#include <boost/asio.hpp>

#include "core/popsink.hpp"

namespace pop
{

#define NETWORK_PACKET_SIZE 368 // in samples
#define NETWORK_STREAM_DATA_TYPE std::complex<float>
#define NETWORK_PACKET_SIZE_BYTES (sizeof(NETWORK_STREAM_DATA_TYPE) * NETWORK_PACKET_SIZE)

	class PopNetworkComplex : public PopSink<NETWORK_STREAM_DATA_TYPE >
	{
	public:
		PopNetworkComplex(const char* incoming_address = "127.0.0.1",
			              int incoming_port = 5004,
			              const char* outgoing_address = "127.0.0.1",
			              int outgoing_port = 5005);
		~PopNetworkComplex();
		void process(const NETWORK_STREAM_DATA_TYPE* data, std::size_t size);

	private:
		void handle_receive(const boost::system::error_code& error,
	      std::size_t /*bytes_transferred*/);
		void start_receive();
		void init();

		boost::asio::ip::udp::socket socket_;
		int incoming_port_;
		int outgoing_port_;

		boost::asio::ip::udp::endpoint incoming_endpoint_;
		boost::asio::ip::udp::endpoint outgoing_endpoint_;
		std::vector<char> recv_buffer_;

		static boost::asio::io_service io_service;
	};
} // namespace pop
