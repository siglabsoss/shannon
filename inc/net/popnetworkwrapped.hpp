#ifndef __POP_NETWORK_WRAPPED__
#define __POP_NETWORK_WRAPPED__

/******************************************************************************
 * Copyright 2013 PopWi Technology Group, Inc. (PTG)
 *
 * This file is proprietary and exclusively owned by PTG or its associates.
 * This document is protected by international and domestic patents where
 * applicable. All rights reserved.
 *
 * This class can send or receive udp packets, or both.  If a sink is connected to this class
 * it will open an outbound network connection.  If a source is connected, it will bind to a udp port.
 *
 * This class drops all timestamps, and only sends bytes from the data circular buffer.  The main usage case
 * is for data that has 1-1 ratio with timestamps.  In this case wrap things up in a boost::pair or a struct.
 *
 * After connecting all sources and sinks, call wakeup() to start networking.
 *
 ******************************************************************************/

#include <boost/asio.hpp>

#include "core/popsink.hpp"
#include "core/popsource.hpp"

#include <iostream>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#define WRAPPED_NETWORK_UDP_PACKET_MTU (1400)

using boost::asio::ip::udp;
using namespace std;



namespace pop
{

typedef struct
{
	boost::asio::ip::address source;
	uint16_t source_port;

} wrapped_net_header_t;
// shared io service across all instances of popnetworkwrapped
extern boost::asio::io_service popnetwork_wrapped_timestamp_io_service;

template <typename DATA_TYPE>
class PopNetworkWrapped : public PopSink<DATA_TYPE>, public PopSource<DATA_TYPE>
{

private:

	boost::asio::ip::udp::socket socket_;
	int incoming_port_;
	int outgoing_port_;
	std::string outgoing_ip_;

	boost::asio::ip::udp::endpoint incoming_endpoint_;
	boost::asio::ip::udp::endpoint outgoing_endpoint_;

	char recv_buffer_[WRAPPED_NETWORK_UDP_PACKET_MTU];



	boost::thread *m_pThread; ///< thead for boost's io_service.run();



public:
	PopNetworkWrapped(int incoming_port, std::string outgoing_ip, int outgoing_port, int chunk = 0)
: PopSink<DATA_TYPE>("PopNetworkWrapped", chunk), PopSource<DATA_TYPE>("PopNetworkWrapped"),
  socket_(popnetwork_wrapped_timestamp_io_service, udp::endpoint(udp::v4(), incoming_port)),
  incoming_port_(incoming_port), outgoing_port_(outgoing_port), outgoing_ip_(outgoing_ip),
  m_pThread(0)
  {

  }
	~PopNetworkWrapped()
	{
	}

	void wakeup()
	{
		if( sizeof(DATA_TYPE) != 1 )
		{
			cout << "Data type is not char or size 1, problems may occur in PopNetworkWrapped" << endl;
		}


//		// if there is a source connected to us
//		if( this->m_rgSource != NULL )
//		{
//			cout << PopSink<DATA_TYPE>::get_name() << " is awake and ready to send." << endl;
//
//			// We assume these were set correctly in the constructor
//			outgoing_endpoint_.address(boost::asio::ip::address::from_string(outgoing_ip_));
//			outgoing_endpoint_.port(outgoing_port_);
//
//			//					cout << this->m_rgSource->get_name() << " is connected to us for input" << endl;
//		}

		// if there is a sink connected to us
		if( this->m_rgSinks.size() != 0 )
		{
			cout << PopSource<DATA_TYPE>::get_name() << " is awake and listening on port " << incoming_port_ << "."<< endl;

			// open the udp port.
			// How is this different than the socket_ constructor in this class's constructor?
			incoming_endpoint_.port(incoming_port_);

			// prep for first packet
			receive_next();

			// start networking run loop in a new thread
			if( 0 == m_pThread )
				m_pThread = new boost::thread(boost::bind(&PopNetworkWrapped::start_io_run_loop, this));

			//					cout << "we connected to " << this->m_rgSinks[0]->get_name() << " for output " << endl;
		}
	}

	void start_io_run_loop()
	{
		// this is blocking
		popnetwork_wrapped_timestamp_io_service.run();
	}


	void init()
	{
	}

	void process(const DATA_TYPE* data, size_t data_size, const PopTimestamp* timestamp_data, size_t timestamp_size)
	{
		if( timestamp_size != 0 )
			cout << "This networking class drops all outgoing timestamps" << endl;

		wrapped_net_header_t header;
		memcpy(&header, data, sizeof(wrapped_net_header_t));

		boost::asio::ip::udp::endpoint destination;
		destination.address(header.source);
		destination.port(header.source_port);

		cout << "destination0: " << endl;
		cout << destination.address().to_string() << endl;


		// calculate how many bytes on the wire
		size_t outputByteCount = (data_size * sizeof(DATA_TYPE)) - sizeof(wrapped_net_header_t);

		// throw warning (TODO this can worked around by sending multiple packets per process)
		if( outputByteCount > WRAPPED_NETWORK_UDP_PACKET_MTU )
			cout << "You asked this networking class to transmit " << outputByteCount << " bytes, but it can't send more than " << WRAPPED_NETWORK_UDP_PACKET_MTU << " at a time." << endl;

		// send packet on the wire
		socket_.send_to(boost::asio::buffer(data+sizeof(wrapped_net_header_t), outputByteCount), destination);

		//				cout << "sending " << outputByteCount << " bytes." << endl;
	}


	void receive_next()
	{
		//				cout << "waiting for next packet" << endl;
		socket_.async_receive_from(
				boost::asio::buffer(recv_buffer_, WRAPPED_NETWORK_UDP_PACKET_MTU), incoming_endpoint_,
				boost::bind(&PopNetworkWrapped::handle_receive, this,
						boost::asio::placeholders::error,
						boost::asio::placeholders::bytes_transferred));
	}

	void handle_receive(const boost::system::error_code& error, std::size_t bytes_transferred)
	{
		//			            cout << "received UDP packet" << endl;

		size_t objectCount = bytes_transferred / sizeof(DATA_TYPE);

		if( bytes_transferred % sizeof(DATA_TYPE) != 0 )
			cout << "received " << bytes_transferred << " bytes which is not evenly divisible by datatype size of " << sizeof(DATA_TYPE) << ".  Discarding the last partial object!" << endl;

		// build
		wrapped_net_header_t header;
		header.source_port = incoming_endpoint_.port();
		header.source = incoming_endpoint_.address();

		// copy of memory allows calling process() only one time
		uint8_t copy[bytes_transferred + sizeof(wrapped_net_header_t)];

		memcpy(copy, &header, sizeof(wrapped_net_header_t));
		memcpy(copy+sizeof(wrapped_net_header_t), recv_buffer_, bytes_transferred);

		// send to our sinks with no timestamps (cuz this class drops them)
		PopSource<DATA_TYPE>::process((DATA_TYPE*) copy, objectCount + sizeof(wrapped_net_header_t), NULL, 0);

		// wait for next packet
		receive_next();
	}

};
} // namespace pop


#undef WRAPPED_NETWORK_UDP_PACKET_MTU

#endif
