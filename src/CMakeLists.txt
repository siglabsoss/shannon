###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################

if(NOT CUDA_FOUND)
    # Set all these cuda variables to empty string so that targets down the line don't freak out
    set(CUDA_CUDART_LIBRARY "")
endif()


set(gpuacc_SRCS
	dsp/prota/popchanfilter.cu)

set(all_SOURCES
	mdl/popradio.cpp
	mdl/poptimestamp.cpp
	mdl/popsymbol.cpp
	core/popassert.c
	core/popsource.cpp
	core/objectstash.cpp
	core/config.cpp
	net/popnetworkcomplex.cpp
	net/popnetworkjson.cpp
	net/popnetworktimestamp.cpp
	net/popnetwork.cpp
	sdr/popuhd.cpp)

set(dsp_SOURCES
	dsp/prota/popchanfilter.cpp
        dsp/prota/popdeconvolve.cpp

)


if(CUDA_FOUND)

cuda_add_library(${LIB_NAME} ${gpuacc_SRCS})




# TEST SHIT Y'ALL
add_executable(test app/test.cpp)
target_link_libraries(test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
set_target_properties(test PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST")
CUDA_ADD_CUFFT_TO_TARGET( test )
add_custom_target(runtest COMMAND test DEPENDS test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})




# For target shannon
add_executable(shannon ${all_SOURCES} ${dsp_SOURCES} app/shannon.cpp)
    
target_link_libraries(shannon ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
CUDA_ADD_CUFFT_TO_TARGET( shannon )




endif() # if CUDA_FOUND



# For target s3p
add_executable(s3p ${all_SOURCES} app/s3p.cpp)
target_link_libraries(s3p ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} uhd)
	




# list of the different files that contain main()
set(executable_list shannon s3p)

# iterate and build each executuable, and add make runx shortcut
foreach(executable ${executable_list}) 
	# if ${executable} is shannon, this creates 'make runshannon'
	add_custom_target(run${executable} COMMAND ${executable} DEPENDS ${executable} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach(executable)


# add target for 'make run' which points at 'make runshannon'
add_custom_target(run COMMAND make runshannon)

