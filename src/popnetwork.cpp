/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include "popnetwork.hpp"

#include <iostream>
#include <stdexcept>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

using boost::asio::ip::udp;
using namespace boost::asio;
using namespace std;

#define NETWORK_PACKET_SIZE 368 // in samples
#define NETWORK_STREAM_DATA_TYPE float
#define NETWORK_BUFFER_SIZE_BYTES (NETWORK_PACKET_SIZE * 100 * sizeof(NETWORK_STREAM_DATA_TYPE))   // in bytes


namespace pop
{

	boost::asio::io_service PopNetwork::io_service;

	PopNetwork::PopNetwork(int incoming_port, int outgoing_port)
	: PopSink<float>(NETWORK_PACKET_SIZE), socket_(io_service, udp::endpoint(udp::v4(), incoming_port)),
	incoming_port_(incoming_port), outgoing_port_(outgoing_port), mp_buf(0),
	m_buf_read_idx(0), m_buf_write_idx(0),
	m_buf_size(NETWORK_BUFFER_SIZE_BYTES / sizeof(NETWORK_STREAM_DATA_TYPE))
	{
		recv_buffer_.resize(12000);

		//start_receive();

		cout << "initialized UDP socket" << endl;

		/* We set the outgoing address on the first incoming packet
		   and set the outgoing port here. */
		outgoing_endpoint_.address(ip::address::from_string("173.167.119.220"));
		outgoing_endpoint_.port(outgoing_port);

		if( NETWORK_BUFFER_SIZE_BYTES % sizeof(NETWORK_STREAM_DATA_TYPE) )
			throw runtime_error("[POPNETWORK] - network stream datatype not divisible by packet length\r\n");

		mp_buf = (uint8_t*)malloc(NETWORK_BUFFER_SIZE_BYTES);
	}

	PopNetwork::~PopNetwork()
	{
		if( mp_buf ) free(mp_buf);
	}

	void PopNetwork::init()
	{
	}

	void PopNetwork::start_receive()
	{
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_), incoming_endpoint_,
			boost::bind(&PopNetwork::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

	void PopNetwork::process(const float* data, size_t len)
	{
		socket_.send_to(boost::asio::buffer(data, NETWORK_PACKET_SIZE * sizeof(float)),
			outgoing_endpoint_);

	}

	void PopNetwork::handle_receive(const boost::system::error_code& error,
      std::size_t /*bytes_transferred*/)
	{
		cout << "received UDP packet" << endl;
		//complex<float> *processed_data;
		if (!error || error == boost::asio::error::message_size)
		{

			// If a callback function exists then process data and send it.
			//send( processed_data, recv_buffer_.size() / sizeof( complex<float> ));

			// receive next packet
			start_receive();
		}
	}
} // namespace pop
