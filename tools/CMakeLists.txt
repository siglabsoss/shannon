###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################

add_executable(validate_implementation src/validate_implementation.cpp)

set_target_properties(validate_implementation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/bin)

target_link_libraries(validate_implementation ${Boost_LIBRARIES} ${CUDA_LIBRARIES} uhd)
