###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################

set(shannon_SOURCES
	shannon.cpp
	popsdr.cpp
	popgpu.cpp
	popnetwork.cpp
	popassert.c)

set(maxwell_SOURCES
	maxwell.cpp
	popsdr.cpp
	popnetwork.cpp
	popassert.c)

set(gpuacc_SRCS
	cudadeconvolve.cu)

cuda_add_library(${LIB_NAME} ${gpuacc_SRCS} )

add_executable(shannon ${shannon_SOURCES})

target_link_libraries(shannon ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd ${LIB_NAME} vrb)

add_executable(maxwell ${maxwell_SOURCES})

target_link_libraries(maxwell ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd ${LIB_NAME} vrb)
