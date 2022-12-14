# Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

IF (WIN32)
cmake_minimum_required (VERSION 3.13) 
ELSE (WIN32)
cmake_minimum_required (VERSION 3.13) 
ENDIF (WIN32)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" ${CMAKE_MODULE_PATH})

#select all sources
FILE(GLOB_RECURSE SRCS "src/*.cu")

add_library(amgx_template_plugin STATIC ${SRCS})

set(AMGX_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../base/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include)

target_sources(amgx_libs PRIVATE ${SRCS})

target_include_directories(amgx_libs PUBLIC ${AMGX_INCLUDES})

target_include_directories(amgx_template_plugin PUBLIC ${THRUST_DIR} ${AMGX_INCLUDES})

target_link_libraries(amgx_template_plugin amgx_libs)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  set_target_properties(amgx_template_plugin PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
endif()

install(TARGETS amgx_template_plugin DESTINATION "lib/sublibs")

