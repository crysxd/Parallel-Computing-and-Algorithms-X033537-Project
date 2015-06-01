#####################################################################
# ClInterface Project
#####################################################################
project(testall)
message(STATUS "Processing ${PROJECT_NAME}")


#####################################################################
# Include the project files
#####################################################################
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)


#####################################################################
# Set to build as an executable
#####################################################################
#add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
sdk_setup_project_bin(${PROJECT_NAME})
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} )
target_link_libraries(${PROJECT_NAME} OpenCL ${GTEST_LIBRARIES} pthread HiddenLayer)
# add_dependencies(${PROJECT_NAME} Core)
# target_link_libraries(${PROJECT_NAME} Core)
