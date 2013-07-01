/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

// std c libaries
#include <iostream>
#include <sstream>
#include <complex>

// boost components
#include <boost/date_time.hpp>
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>

// USRP components
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>

// shannon components
#include "popsdr.hpp"


using namespace std;


namespace pop
{

	#define NUM_RX_BUFS 1000

	const char *cuda_error[] = {
		"[ERROR_CODE_NONE] - No error assoiciated with this metadata.",
		"[ERROR_CODE_TIMEOUT] - No packet received, implementation timed-out.",
		"[ERROR_CODE_LATE_COMMAND] - A stream command was issued in the past.",
		0,
		"[ERROR_CODE_BROKEN_CHAIN] - Expected another stream command.",
		0,
		0,
		0,
		"[ERROR_CODE_OVERFLOW] - An internal receive buffer has filled.",
		0,
		0,
		0,
		"[ERROR_CODE_ALIGNMENT] - Multi-channel alignment failed.",
		0,
		0,
		"[ERROR_CODE_BAD_PACKET] - The packet could not be parsed.",
		0
	};

	/**
	 * Constructor for Software Defined radio class.
	 */
	PopSdr::PopSdr() : mp_thread(0)
	{
		start();
	}


	/**
	 * Destructor for Software Defined radio class.
	 */
	PopSdr::~PopSdr()
	{
		// if thread is still running then shut it down
		// TODO
	}


	/**
	 * This is the actual process I/O loop for the SDR. This should
	 * run in its own thread. This only stops if it is commanded to
	 * do so or on error.
	 */
	POP_ERROR PopSdr::run()
	{
        size_t num_rx_samps;
        unsigned rx_buf_idx = 0;


        /* This will fail unless you have sudo permissions but its ok.
           Giving UHD thread priority control can reduce overflows.*/
        uhd::set_thread_priority_safe();

        // create device (TODO: don't know if this is best method)
        usrp = uhd::usrp::multi_usrp::make(std::string());
        std::cout << boost::format("Using Device: %s") %
            usrp->get_pp_string() << std::endl;

        usrp->set_rx_freq(902840000);
        usrp->set_rx_rate(1e6);

        usrp->set_time_source("external");

        // synchronize time across all motherboards (2 seconds to complete)
        usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));
        boost::this_thread::sleep(boost::posix_time::seconds(1));

        //create a receive streamer
        //linearly map channels (index0 = channel0, index1 = channel1, ...)
        uhd::stream_args_t stream_args("fc32"); //complex floats
        for (size_t chan = 0; chan < usrp->get_rx_num_channels(); chan++)
            stream_args.channels.push_back(chan); //linear mapping

        // create the RX stream
        rx_stream = usrp->get_rx_stream(stream_args);

        //setup streaming
        uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        stream_cmd.time_spec = uhd::time_spec_t(1.5);
        usrp->issue_stream_cmd(stream_cmd); //tells all channels to stream

        //allocate buffers to receive with samples (one buffer per channel)
        const size_t samps_per_buff = rx_stream->get_max_num_samps();
        std::vector<std::vector<std::complex<float> > > buffs( usrp->get_rx_num_channels(), std::vector<std::complex<float> >(samps_per_buff * NUM_RX_BUFS) );

        //create a vector of pointers to point to each of the channel buffers
        std::vector<std::vector<std::complex<float> *> > buff_ptrs(NUM_RX_BUFS, std::vector<std::complex<float> *>(buffs.size(),0));
        for( size_t j = 0; j < NUM_RX_BUFS; j++ )
        {
        	for (size_t i = 0; i < buffs.size(); i++)
    		{
    			buff_ptrs[j][i] = &buffs[i].front();
    		}
        }
        

        //the first call to recv() will block this many seconds before receiving
        double timeout = 1.5 + 0.1; //timeout (delay before receive + padding)

		for(;;)
		{
            //receive a single packet
            num_rx_samps = rx_stream->recv(buff_ptrs[rx_buf_idx], samps_per_buff, md,
                                           timeout);

            //use a small timeout for subsequent packets
            timeout = 0.1;

            //handle the error code
            if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) break;
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE)
            {
                throw std::runtime_error(str(boost::format(
                    "Unexpected error code 0x%x - %s"
                ) % md.error_code % cuda_error[md.error_code]));
            }

            // wrap around rx buffer index
            rx_buf_idx = (rx_buf_idx + 1) % NUM_RX_BUFS;

            // if rx buffer full then dump data
            if( 0 == rx_buf_idx )
            {
            	sig(buff_ptrs[0][0], num_rx_samps *
            	    NUM_RX_BUFS );
            }
		}

		return POP_ERROR_UNKNOWN; // it should never actually get here
	}


	/**
	 * Connects subscriber to SDR data source
	 * @param func Callback function for passing data.
	 */
	POP_ERROR PopSdr::connect(SDR_DATA_FUNC func)
	{
		sig.connect(func);

		return POP_ERROR_NONE;
	}


	/**
	 * Start sampling RF data. This calls its own thread.
	 */
	POP_ERROR PopSdr::start()
	{
		// check to see if thread is already running for this object
		if( mp_thread ) return POP_ERROR_ALREADY_RUNNING;

        if( usrp ) return POP_ERROR_ALREADY_RUNNING;

		// create a new threat that runs object's process I/O loop
		mp_thread = new boost::thread(boost::bind(&PopSdr::run, this));

		// if thread was not created return an error
		if( 0 == mp_thread ) return POP_ERROR_UNKNOWN;

		return POP_ERROR_NONE;
	}


	/**
	 * Stop sampling RF data.
	 */
	POP_ERROR PopSdr::stop()
	{
		return POP_ERROR_NONE;
	}
}
