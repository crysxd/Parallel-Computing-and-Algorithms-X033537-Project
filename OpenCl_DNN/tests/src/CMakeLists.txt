cmake_minimum_required(VERSION 2.6)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(../../src/activations)
include_directories(../../src/cl_prog)
include_directories(../../src/cost)
include_directories(../../src/hiddenlayer)
include_directories(../../src/nnet)
include_directories(../../src/cl_intf)
# Link runTests with what we want to test and the GTest and pthread library
add_executable(runtest testall.cpp)
add_dependencies(runtest Activation.h)
target_link_libraries(runtest ${GTEST_LIBRARIES} pthread)
