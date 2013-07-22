###############################################################################
# Copyright 2013 PopWi Technology Group, Inc. (PTG)
#
# This file is proprietary and exclusively owned by PTG or its associates.
# This document is protected by international and domestic patents where
# applicable. All rights reserved.
#
###############################################################################

# DLMALLOC is a general 6x improvement on GCC memory allocation.

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -DREPLACE_SYSTEM_ALLOCATOR")
add_library(nedmalloc nedmalloc.c)
