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

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

using boost::asio::ip::udp;
using namespace std;

namespace pop
{

	boost::asio::io_service PopNetwork::io_service;

	PopNetwork::PopNetwork(int incoming_port, int outgoing_port)
	: socket_(io_service, udp::endpoint(udp::v4(), incoming_port)),
	incoming_port_(incoming_port), outgoing_port_(outgoing_port),
	outgoing_address_is_set_(false), callback_(0)
	{
		recv_buffer_.resize(12000);

		start_receive();

		cout << "initialized UDP socket" << endl;

		/* We set the outgoing address on the first incoming packet
		   and set the outgoing port here. */
		outgoing_endpoint_.port(outgoing_port);
	}

	void PopNetwork::start_receive()
	{
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_), incoming_endpoint_,
			boost::bind(&PopNetwork::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

	void PopNetwork::send(void* data, std::size_t size)
	{
		socket_.send_to(boost::asio::buffer(data, size), outgoing_endpoint_);
	}

	void PopNetwork::handle_receive(const boost::system::error_code& error,
      std::size_t /*bytes_transferred*/)
	{
		cout << "received UDP packet" << endl;
		void *processed_data;
		if (!error || error == boost::asio::error::message_size)
		{
    		/* the first time we receive a message let that be the
               outgoing address for all future packets */
			if( false == outgoing_address_is_set_ )
			{
				outgoing_endpoint_.address(incoming_endpoint_.address());
				outgoing_address_is_set_ = true;
			}

			// If a callback function exists then process data and send it.
			if( callback_ )
			{
				processed_data = callback_( &recv_buffer_[0], recv_buffer_.size());
				send( processed_data, recv_buffer_.size());
			}

			// receive next packet
			start_receive();
		}
	}
} // namespace pop
