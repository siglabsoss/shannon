###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################


set(gpuacc_SRCS
	dsp/prota/popchanfilter.cu)

set(shannon_SOURCES
	mdl/popradio.cpp
	app/shannon.cpp
	core/popassert.c
	dsp/prota/popchanfilter.cpp
	dsp/prota/popdeconvolve.cpp
	net/popnetworkcomplex.cpp
	net/popnetworkjson.cpp
	net/popnetwork.cpp
	sdr/popuhd.cpp)

cuda_add_library(${LIB_NAME} ${gpuacc_SRCS} )

add_executable(shannon ${shannon_SOURCES})

target_link_libraries(shannon ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
CUDA_ADD_CUFFT_TO_TARGET( shannon )

add_custom_target(run COMMAND shannon DEPENDS shannon WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
