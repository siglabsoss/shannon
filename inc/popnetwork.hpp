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

#include <popsink.hpp>

namespace pop
{
	class PopNetwork : public PopSink<float>
	{
	public:
		PopNetwork(int incoming_port = 5004, int outgoing_port = 35005);
		~PopNetwork();
		void process(float* data, std::size_t size);

	private:
		void handle_receive(const boost::system::error_code& error,
	      std::size_t /*bytes_transferred*/);
		void start_receive();
		void init();

		boost::asio::ip::udp::socket socket_;
		int incoming_port_;
		int outgoing_port_;

		uint8_t *mp_buf; ///< circular buffer
		size_t m_buf_read_idx; ///< read index in samples
		size_t m_buf_write_idx; ///< write index in samples
		size_t m_buf_size; ///< circular buffer size in samples

		boost::asio::ip::udp::endpoint incoming_endpoint_;
		boost::asio::ip::udp::endpoint outgoing_endpoint_;
		std::vector<char> recv_buffer_;

		static boost::asio::io_service io_service;
	};
} // namespace pop
