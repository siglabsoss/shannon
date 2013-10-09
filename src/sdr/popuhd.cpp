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

// PopWi components
#include "sdr/popuhd.hpp"


using namespace std;


namespace pop
{

	#define NUM_RX_BUFS 100

	const char *uhd_error[] = {
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
	PopUhd::PopUhd() : PopSource<>("PopUhd"), mp_thread(0), m_timestamp_offset(0, 0.0)
	{
	}


	/**
	 * Destructor for Software Defined radio class.
	 */
	PopUhd::~PopUhd()
	{
		// if thread is still running then shut it down
		// TODO
	}

	// code pulled from '/home/joel/uhd/host/lib/types/time_spec.cpp
	// because that file was compiled with incorrect flags and get_system_time() returns garbage
	namespace pt = boost::posix_time;
	uhd::time_spec_t get_microsec_system_time(void){
	    pt::ptime time_now = pt::microsec_clock::universal_time();
	    pt::time_duration time_dur = time_now - pt::from_time_t(0);
	    return uhd::time_spec_t(
	        time_t(time_dur.total_seconds()),
	        long(time_dur.fractional_seconds()),
	        double(pt::time_duration::ticks_per_second())
	    );
	}




	/**
	 * This is the actual process I/O loop for the SDR. This should
	 * run in its own thread. This only stops if it is commanded to
	 * do so or on error.
	 */
	POP_ERROR PopUhd::run()
	{
        /* This will fail unless you have sudo permissions but its ok.
           Giving UHD thread priority control can reduce overflows.*/
        uhd::set_thread_priority_safe();

        // create device (TODO: don't know if this is best method)
        usrp = uhd::usrp::multi_usrp::make(std::string());
        std::cout << boost::format("Using Device: %s") %
            usrp->get_pp_string() << std::endl;

        usrp->set_rx_freq(POP_PROTA_BLOCK_A_UPLK);
        usrp->set_rx_rate(POP_PROTA_BLOCK_A_WIDTH);
        usrp->set_rx_antenna("RX2");
        usrp->set_rx_gain(25);

        vector<string> vstr;
        vector<string>::iterator vstrit;

        vstr = usrp->get_mboard_sensor_names();

        for( vstrit = vstr.begin(); vstrit != vstr.end(); vstrit++ )        	
        	cout << "sensor names: " << *vstrit << endl;

        double actual_rate = usrp->get_rx_rate();

        std::cout << "actual RX sample rate: " << actual_rate << "Hz" << std::endl;

#ifndef OPTION_DISABLE_GPS
        usrp->set_time_source("external");

        // synchronize time across all motherboards (2 seconds to complete)
        usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));
        boost::this_thread::sleep(boost::posix_time::seconds(1));
#endif

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

        size_t samps_received;

        //the first call to recv() will block this many seconds before receiving
        double timeout = 1.5 + 0.1; //timeout (delay before receive + padding)

		for(;;)
		{
			std::vector<std::complex<float> *> buf;

			buf.push_back(get_buffer(samps_per_buff));

            //receive a single packet, TODO confirm num samps received
            samps_received = rx_stream->recv(buf, samps_per_buff, md, timeout);

            // wait till the uhd timestamp rolls over to the next second
            if( md.time_spec.get_frac_secs() < .0009 && m_timestamp_offset.get_full_secs() == 0 )
            {
            	// sample system time
                uhd::time_spec_t now = get_microsec_system_time();

                // round system time to nearest second
                m_timestamp_offset = uhd::time_spec_t(round(now.get_real_secs()));

                // below we add the radio seconds (which count up since launch) to our offset which doesn't change.
                // at this point the radio seconds are probably about 2.0001
                // we want to subract the whole seconds from our m_timestamp_offset right now (one time) so we can just do a simple add below and get real time
				// ( this uses the constructor to construct a temporary timestamp object holding just N whole seconds. then we use the -= overload to subtract it)
                m_timestamp_offset -= uhd::time_spec_t(md.time_spec.get_full_secs());


//                cout << "rounded to base: '" << m_timestamp_offset.get_full_secs() << "' '" <<  m_timestamp_offset.get_frac_secs()<< "' from now of: '" << now.get_full_secs()  << "' '" << now.get_frac_secs() << "'" << endl;

            }

            // build a pop timestamp from uhd time + offset.
            // the time always applies to sample 0
            PopTimestamp pop_stamp = PopTimestamp(md.time_spec + m_timestamp_offset, 0);

            // cout << "    samp time was" << boost::lexical_cast<string>(pop_stamp.get_real_secs()) << endl;

            //use a small timeout for subsequent packets
            timeout = 0.1;

            //handle the error code
            if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) break;
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE)
            {
                throw std::runtime_error(str(boost::format(
                    "Unexpected error code 0x%x - %s"
                ) % md.error_code % uhd_error[md.error_code]));
            }

            // only process() if m_timestamp_offset is valid
            if( m_timestamp_offset.get_full_secs() != 0 )
            {
            	// process new data in source
            	process(samps_received, &pop_stamp, 1);
            }
		}

		return POP_ERROR_UNKNOWN; // it should never actually get here
	}



	/**
	 * Start sampling RF data. This calls its own thread.
	 */
	POP_ERROR PopUhd::start()
	{
		// check to see if thread is already running for this object
		if( mp_thread ) return POP_ERROR_ALREADY_RUNNING;

        if( usrp ) return POP_ERROR_ALREADY_RUNNING;

		// create a new threat that runs object's process I/O loop
		mp_thread = new boost::thread(boost::bind(&PopUhd::run, this));

		// if thread was not created return an error
		if( 0 == mp_thread ) return POP_ERROR_UNKNOWN;

		return POP_ERROR_NONE;
	}


	/**
	 * Stop sampling RF data.
	 */
	POP_ERROR PopUhd::stop()
	{
		return POP_ERROR_NONE;
	}
}
