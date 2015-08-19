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

cmake_minimum_required(VERSION 2.8)

# These enable Arm crosscompiler
#set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
#set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")

project(shannon CXX C)

# What does this do? Do we need it?
#set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/3rd/boost)
#set(Boost_NO_SYSTEM_PATHS true)
#set(Boost_NO_BOOST_CMAKE true)
#set(Boost_USE_STATIC_LIBS ON)

set(shannon_VERSION_MAJOR 0)
set(shannon_VERSION_MINOR 1)
set(shannon_VERSION_PATCH 0)

set(shannon_VERSION ${shannon_VERSION_MAJOR}.${shannon_VERSION_MINOR}.${shannon_VERSION_PATCH})

set(CMAKE_BUILD_TYPE debug)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(Boost COMPONENTS date_time thread filesystem program_options prg_exec_monitor regex system unit_test_framework)


set(LIB_NAME "gpuacceleration")

#  OPTIONS -DSTUFF="blah blah"
#  RELEASE -DNDEBUG
#  DEBUG -g -DDEBUG


configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_SOURCE_DIR}/Doxyfile @ONLY)

add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)

########################################################################
# Set the compiler and linker flags
########################################################################
set(POP_ALL_FLAGS "-DDEBUG -Wall -Wno-switch -Wno-unused-function -Wno-unused-variable -D__STDC_FORMAT_MACROS")
set(POP_CXX_FLAGS "-std=c++0x")
set(POP_C_FLAGS "")
set(POP_EXE_LINKER_FLAGS "")

set(CMAKE_CXX_FLAGS "${CMAKE_CCX_FLAGS} ${POP_ALL_FLAGS} ${POP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}	${POP_ALL_FLAGS} ${POP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${POP_EXE_LINKER_FLAGS}")


########################################################################
# Setup the include and linker paths
########################################################################
include_directories(
	${CMAKE_SOURCE_DIR}/src/mdl
	${CMAKE_SOURCE_DIR}/inc
	${CMAKE_SOURCE_DIR}/src
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
add_subdirectory(3rd)
add_subdirectory(doc)
add_subdirectory(tools)
