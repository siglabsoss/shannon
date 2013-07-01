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
	popgpu.cpp
	popsdr.cpp
	popcontrol.cpp
	popnetwork.cpp)

set(gpuacc_SRCS
	cudadeconvolve.cu)

cuda_add_library(${LIB_NAME}
  ${gpuacc_SRCS} )

add_executable(shannon ${shannon_SOURCES})

target_link_libraries(shannon ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
