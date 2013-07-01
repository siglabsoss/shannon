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

namespace pop
{
	typedef void*(*POP_GPU_CALLBACK)(void*,std::size_t);

	class PopNetwork
	{
	public:
		PopNetwork(int incoming_port = 5004, int outgoing_port = 5005);
		static void init();
		void send(float* data, std::size_t size);

	private:
		void handle_receive(const boost::system::error_code& error,
	      std::size_t /*bytes_transferred*/);
		void start_receive();

		boost::asio::ip::udp::socket socket_;
		boost::asio::ip::udp::endpoint incoming_endpoint_;
		boost::asio::ip::udp::endpoint outgoing_endpoint_;
		std::vector<char> recv_buffer_;
		POP_GPU_CALLBACK callback_;
		bool outgoing_address_is_set_;
		int incoming_port_;
		int outgoing_port_;

		static boost::asio::io_service io_service;
	};
} // namespace pop
