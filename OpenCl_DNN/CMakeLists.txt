cmake_minimum_required(VERSION 2.6)




################################################################################
#                            GLOBAL CONFIGURATION                              #
################################################################################

# Project name
project(OpenCLDNN)

# Static library name
set(DNN_STATIC_LIB opcldnn)

# Executable name
set(EXECUTEABLE run)

# Path to the executeable
set(EXECUTEABLE_PATH ${PROJECT_SOURCE_DIR}/src/main.cpp)

#Test executeable name
set(EXECUTEABLE_TEST test)

#Path to Main test
set(EXECUTEABLE_TEST_PATH ${PROJECT_SOURCE_DIR}/tests/testall.cpp)

# TestName
set(EXECUTEABLE_TEST test)

# Binary directory
set(BINARY_DIR ${CMAKE_SOURCE_DIR}/bin/)

#Add for OpenCL the cmake module
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

################################################################################
#                             COMPILATION FLAGS                                #
################################################################################

# Common flags

add_definitions(-DDEBUG=0)
add_definitions(-DNO_OUTPUT=0)
add_definitions(-DCL_SOURCE_DIR="${CMAKE_SOURCE_DIR}/src/")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # C++11 flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

    # Base compiler flags
    # https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    # https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
    # http://stackoverflow.com/a/9862800
    set(CXX_BASE_FLAGS
        "-W -Wall -Wextra -Weffc++ -pedantic"
        "-Wold-style-cast -Wparentheses -Wlogical-op -Woverloaded-virtual"
        "-Wfloat-equal -Wpointer-arith -Wcast-align -Wcast-qual"
        "-Winit-self -Wwrite-strings -Wstrict-null-sentinel"
        "-Wredundant-decls -Wswitch-default -Wundef"
        "-Wmissing-include-dirs"
        "-Wdisabled-optimization"
        #-Wunused-parameter
        #"-Werror"
        #"-Wnoexcept"

        #"-Wshadow"
        #"-Wsign-conversion"
        #"-Wconversion"
        #"-Wstrict-overflow=5"
        #"-Winline"
        #"-Wpadded"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    # set(CXX_FLAGS_RELEASE "-Werror ${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CXX_DEFINES} ${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS_RELEASE}")

    # Debug mode
    set(CXX_FLAGS_DEBUG ${CXX_BASE_FLAGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CXX_DEFINES} ${CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS_DEBUG}")

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )

elseif(${MSYS})

    # Base compiler flags
    set(CXX_BASE_FLAGS
        "/MP4"
        "/W0"
        #"/Wall"
    )
    string(REGEX REPLACE ";" " " CXX_BASE_FLAGS "${CXX_BASE_FLAGS}")

    # Release mode
    set(CXX_FLAGS_RELEASE ${CXX_BASE_FLAGS})
    set(CXX_FLAGS_RELEASE ${CXX_FLAGS_RELEASE} "/O2")
    string(REGEX REPLACE ";" " " CXX_FLAGS_RELEASE "${CXX_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS_RELEASE}")

    # Debug mode
    set(CXX_FLAGS_DEBUG ${CXX_BASE_FLAGS})
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS_DEBUG}")

    # Release with debug symbols
    set(CXX_FLAGS_RELWITHDEBINFO "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_FLAGS_RELWITHDEBINFO}"
    )

    # Release with stripping of extra symbols
    set(CXX_FLAGS_MINSIZEREL "${CXX_BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} ${CXX_FLAGS_MINSIZEREL}"
    )


    # Remove SL security warnings with Visual C++
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)

else()

    message(FATAL_ERROR "Unsupported compiler")
    return()

endif()


################################################################################
#                                 CUSTOM COMMANDS                              #
################################################################################

if(OS_LINUX)

    # Clean up all generated files
    add_custom_target(
        trash
        COMMAND rm -r ${CMAKE_SOURCE_DIR}/bin/* ${CMAKE_SOURCE_DIR}/build/*
        COMMENT "Deleting generated files"
    )

    # Delete generated files and reset project settings
    add_custom_target(
        reset
        COMMAND make trash\; cmake ..
        COMMENT "Resetting project settings"
    )

endif()


################################################################################
#                                   EXECUTABLE                                 #
################################################################################

set(ACTIVATIONS_DIR src/activations)
set(CL_INTERFACE_DIR src/cl_intf)
set(COST_DIR src/cost)
set(HIDDENLAYER_DIR src/hiddenlayer)
set(NNET_DIR src/nnet)
set(MATH_DIR src/math)
set(WRAPPER_DIR src/wrapper)

# Find necessary packages
find_package(GTest REQUIRED)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
# set(LIBS ${LIBS} ${OpenCL_LIBRARIES})

# for example, on Fedora with beignet, use:
# export CUSTOM_OPENCL=/usr/lib64/beignet/libcl.so
if(DEFINED ENV{CUSTOM_OPENCL})
    set(OPENCL $ENV{CUSTOM_OPENCL})
else()
    set(OPENCL OpenCL)
endif()

# Include directories
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${ACTIVATIONS_DIR} ${CL_INTERFACE_DIR} ${COST_DIR} ${HIDDENLAYER_DIR} ${NNET_DIR} ${MATH_DIR})



# Source files
set(HEADERS
    ${ACTIVATIONS_DIR}/Activation.h
    ${ACTIVATIONS_DIR}/Sigmoid.h
    ${ACTIVATIONS_DIR}/TanH.h
    ${CL_INTERFACE_DIR}/OpenCL.hpp
    ${CL_INTERFACE_DIR}/util.hpp
    ${CL_INTERFACE_DIR}/OpenCLPort.h
    ${COST_DIR}/Cost.h
    ${COST_DIR}/MSE.h
    ${HIDDENLAYER_DIR}/HiddenLayer.h
    ${NNET_DIR}/FeedForwardNN.h
    ${MATH_DIR}/CLMatrix.hpp
)

# Source files
set(SOURCES
    ${ACTIVATIONS_DIR}/Activation.cpp
    ${ACTIVATIONS_DIR}/Sigmoid.cpp
    ${ACTIVATIONS_DIR}/TanH.cpp
    ${CL_INTERFACE_DIR}/util.cpp
    ${CL_INTERFACE_DIR}/OpenCLPort.cpp
    ${COST_DIR}/Cost.cpp
    ${COST_DIR}/MSE.cpp
    ${HIDDENLAYER_DIR}/HiddenLayer.cpp
    ${NNET_DIR}/FeedForwardNN.cpp
)

# Static library
set(LIBRARY_OUTPUT_PATH ${BINARY_DIR})
add_library(${DNN_STATIC_LIB}
    STATIC ${HEADERS} ${SOURCES}
)


# FIND_LIBRARY(OPENCL_LIBRARY cl /usr/lib64/beignet)

# Executable for the application
set(EXECUTABLE_OUTPUT_PATH ${BINARY_DIR})
add_executable(${EXECUTEABLE} ${EXECUTEABLE_PATH})
target_link_libraries(${EXECUTEABLE} ${DNN_STATIC_LIB} ${OPENCL} pthread)
#Executeable for the tests

add_executable(${EXECUTEABLE_TEST} ${EXECUTEABLE_TEST_PATH})
target_link_libraries(${EXECUTEABLE_TEST} ${DNN_STATIC_LIB} ${GTEST_LIBRARIES} ${OPENCL} pthread)

################################################################################
#                                 PYTHON WRAPPER                               #
################################################################################

add_library(nn SHARED ${WRAPPER_DIR}/NeuralNetwork.cpp ${SOURCES})
target_link_libraries(nn ${OPENCL} pthread)
