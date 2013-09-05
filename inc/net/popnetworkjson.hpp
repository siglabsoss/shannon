#ifndef __POP_NETWORK_JSON_
#define __POP_NETWORK_JSON_

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
#include "core/popsource.hpp"

#define POPNETWORK_SOCKET_MAXLEN (1024)

namespace pop
{
	class PopNetworkJson : public PopSink<float>, public PopSource<float>
	{
	public:
		PopNetworkJson(int incoming_port = 5004, int outgoing_port = 35005);
		~PopNetworkJson();
		void process(const float* data, std::size_t size);

	private:
                // forward declare stuff
		void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred);
		void start_receive();
		void init();
		void thread_run();
		void test();
		size_t get_length(uint8_t* data, size_t max_length);
		size_t get_length_from_header(uint8_t* data, size_t max_length);
		void build_message_header(const size_t byte_count, char header[3], size_t* header_len);
		void handle_json(uint8_t* bytes, size_t len);

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

		static boost::asio::io_service io_service;
                
                boost::thread *m_pThread; ///< thead for boost's io_service.run();
                
                
	};
} // namespace pop

#endif
