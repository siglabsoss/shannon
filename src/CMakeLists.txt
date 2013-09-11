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

set(all_SOURCES
	mdl/popradio.cpp
	core/popassert.c
	core/popsource.cpp
	core/objectstash.cpp
	core/config.cpp
	dsp/prota/popchanfilter.cpp
	dsp/prota/popdeconvolve.cpp
	net/popnetworkcomplex.cpp
	net/popnetworkjson.cpp
	net/popnetwork.cpp
	sdr/popuhd.cpp)

cuda_add_library(${LIB_NAME} ${gpuacc_SRCS})



# list of the different files that contain main()
set(executable_list shannon s3p ben)


# TEST SHIT Y'ALL
add_executable(test app/test.cpp)
target_link_libraries(test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
set_target_properties(test PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST")
CUDA_ADD_CUFFT_TO_TARGET( test )
add_custom_target(runtest COMMAND test DEPENDS test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})



# iterate and build each executuable
foreach(executable ${executable_list}) 
    #message("Building " ${executable} "...")
    
    add_executable(${executable} ${all_SOURCES} app/${executable}.cpp)
    
    target_link_libraries(${executable} ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
	CUDA_ADD_CUFFT_TO_TARGET( ${executable} )
	
	# if ${executable} is shannon, this creates 'make runshannon'
	add_custom_target(run${executable} COMMAND ${executable} DEPENDS ${executable} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

 
endforeach(executable)

# add target for 'make run' which points at 'make runshannon'
add_custom_target(run COMMAND make runshannon)



