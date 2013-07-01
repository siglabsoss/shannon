###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################



########################################################################
# Project setup
########################################################################
cmake_minimum_required(VERSION 2.6)
project(shannon CXX C)
set(shannon_VERSION_MAJOR 0)
set(shannon_VERSION_MINOR 1)

set(CMAKE_BUILD_TYPE debug)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_library(UHD_LIBRARY uhd)

find_package(Boost 1.48.0 COMPONENTS date_time thread filesystem
	timer program_options prg_exec_monitor regex system)

find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA Library not properly installed")
endif()

set(CUDA_64_BIT_DEVICE_CODE ON)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_VERBOSE_BUILD ON)
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS -arch=sm_20)

set(LIB_NAME "gpuacceleration")

#  OPTIONS -DSTUFF="blah blah"
#  RELEASE -DNDEBUG
#  DEBUG -g -DDEBUG


########################################################################
# Setup the include and linker paths
########################################################################
include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/3rd
    ${Boost_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${Boost_LIBRARY_DIRS}
    ${CUDA_LIBRARY_DIRS}
)

########################################################################
# Add subdirectories
########################################################################
add_subdirectory(src)	
add_subdirectory(docs)

