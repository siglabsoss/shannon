###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################

if(CUDA_FOUND)

add_executable(validate_implementation EXCLUDE_FROM_ALL src/validate_implementation.cpp)

set_target_properties(validate_implementation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/bin)

target_link_libraries(validate_implementation ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd)



add_executable(usb_reset EXCLUDE_FROM_ALL src/usb_reset.c)

set_target_properties(usb_reset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/bin)


endif() # if CUDA_FOUND
