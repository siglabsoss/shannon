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
#include <complex>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include "net/popnetworkcomplex.hpp"

using boost::asio::ip::udp;
using namespace boost::asio;
using namespace std;

namespace pop
{

	boost::asio::io_service PopNetworkComplex::io_service;

	PopNetworkComplex::PopNetworkComplex(const char* incoming_address, int incoming_port, const char* outgoing_address, int outgoing_port)
	: PopSink<NETWORK_STREAM_DATA_TYPE >("PopNetworkComplex", NETWORK_PACKET_SIZE), socket_(io_service, udp::endpoint(udp::v4(), incoming_port)),
	incoming_port_(incoming_port), outgoing_port_(outgoing_port)
	{
		cout << "initialized UDP socket" << endl;

		/* We set the outgoing address on the first incoming packet
		   and set the outgoing port here. */
		outgoing_endpoint_.address(ip::address::from_string(outgoing_address));
		outgoing_endpoint_.port(outgoing_port);
	}

	PopNetworkComplex::~PopNetworkComplex()
	{
	}

	void PopNetworkComplex::init()
	{
	}

	void PopNetworkComplex::start_receive()
	{
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_), incoming_endpoint_,
			boost::bind(&PopNetworkComplex::handle_receive, this,
				boost::asio::placeholders::error,
				boost::asio::placeholders::bytes_transferred));
	}

	void PopNetworkComplex::process(const NETWORK_STREAM_DATA_TYPE* data, size_t len, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		socket_.send_to(boost::asio::buffer(data, NETWORK_PACKET_SIZE_BYTES),
			outgoing_endpoint_);
	}

	void PopNetworkComplex::handle_receive(const boost::system::error_code& error,
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
