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
#include <string>

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include "net/popnetworktimestamp.hpp"
#include "json/json.h"
#include "core/utilities.hpp"





namespace pop
{

	boost::asio::io_service popnetwork_timestamp_io_service;
        
} // namespace pop
