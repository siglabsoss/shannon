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

# The subset of source files that all apps share
set(shared_SOURCES
	mdl/popradio.cpp
	mdl/poptimestamp.cpp
	core/popassert.c
	core/popsource.cpp
	core/objectstash.cpp
	core/config.cpp
	core/utilities.cpp
	net/popnetwork.cpp
	../3rd/frozen/frozen.c
	../3rd/b64/b64.c
	core/popjsonrpc.cpp
	core/popchannelmap.cpp
	core/popfabric.cpp
	../3rd/boost_serial/AsyncSerial.cpp
	core/geohelper.cpp
)

# S3P and Shannon sources
set(all_SOURCES
	${shared_SOURCES}
	mdl/popsymbol.cpp
	mdl/poppeak.cpp
	core/basestationfreq.c
	net/popnetworkcomplex.cpp
	net/popnetworkjson.cpp
	core/popbuchermultilateration.cpp
	core/popcoordinatetransform.cpp
	core/popfangmultilateration.cpp
	core/popgeolocation.cpp
	core/popgravitinoparser.cpp
	core/popsightingstore.cpp
	core/poptrackerlocationstore.cpp
	net/popwebhook.cpp
	dsp/prota/popsparsecorrelate.c
)

# Gravatinio only sources
set(gravatino_SOURCES
   ${shared_SOURCES}
	core/popparsegps.cpp
	core/popgpsdevice.cpp
	core/popartemisrpc.cpp
	core/pops3prpc.cpp
	core/poppackethandler.cpp
	core/popserial.cpp
	core/popfabricbridge.cpp
	dsp/prota/popsparsecorrelate.c
)

set(dsp_SOURCES
	dsp/prota/popchanfilter.cpp
	dsp/prota/popdeconvolve.cpp
	dsp/prota/popbinner.cpp
)

set(uhd_SOURCES
	sdr/popuhd.cpp	
)


# Helper to mark target as having uhd library so source can handle this
# This MUST be called or ommited for every target correctly
# See http://www.kitware.com/blog/home/post/390 for an alternate way to inspect target properties
function(has_uhd tgt)
  set_target_properties(${tgt} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DTARGET_HAS_UHD")
endfunction()


if(CUDA_FOUND)

cuda_add_library(${LIB_NAME} ${gpuacc_SRCS})




# TEST SHIT Y'ALL
add_executable(test app/test.cpp)
target_link_libraries(test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
set_target_properties(test PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST -DTARGET_HAS_UHD")
CUDA_ADD_CUFFT_TO_TARGET( test )
add_custom_target(runtest COMMAND test DEPENDS test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})




# For target shannon
add_executable(shannon ${all_SOURCES} ${dsp_SOURCES} ${uhd_SOURCES} app/shannon.cpp)
has_uhd(shannon)
target_link_libraries(shannon ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd ${LIB_NAME})
CUDA_ADD_CUFFT_TO_TARGET( shannon )




endif() # if CUDA_FOUND



# For target s3p
add_executable(s3p ${all_SOURCES} app/s3p.cpp)
target_link_libraries(s3p ${Boost_LIBRARIES} ${ZEROMQ_LIBRARY})

# For target gravitino
add_executable(gravitino ${gravatino_SOURCES} app/gravitino.cpp)
target_link_libraries(gravitino ${Boost_LIBRARIES} ${ZEROMQ_LIBRARY})
	


# MORE TESTS
add_executable(geohelper_test ${all_SOURCES} app/geohelper_test.cpp)
target_link_libraries(geohelper_test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ZEROMQ_LIBRARY})

add_executable(popbuchermultilateration_test ${all_SOURCES} app/popbuchermultilateration_test.cpp)
target_link_libraries(popbuchermultilateration_test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ZEROMQ_LIBRARY})

add_executable(popcoordinatetransform_test ${all_SOURCES} app/popcoordinatetransform_test.cpp)
target_link_libraries(popcoordinatetransform_test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ZEROMQ_LIBRARY})

add_executable(popgeolocation_test ${all_SOURCES} app/popgeolocation_test.cpp)
target_link_libraries(popgeolocation_test ${Boost_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ZEROMQ_LIBRARY})



# list of the different files that contain main()
set(executable_list shannon s3p gravitino geohelper_test popbuchermultilateration_test popcoordinatetransform_test popgeolocation_test)

# iterate and build each executuable, and add make runx shortcut
foreach(executable ${executable_list}) 
	# if ${executable} is shannon, this creates 'make runshannon'
	add_custom_target(run${executable} COMMAND ${executable} DEPENDS ${executable} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach(executable)


# add target for 'make run' which points at 'make runshannon'
add_custom_target(run COMMAND make runshannon)

