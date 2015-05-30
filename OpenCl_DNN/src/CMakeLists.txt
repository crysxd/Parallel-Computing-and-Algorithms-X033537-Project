## Define Project####
project (DNN)
message(STATUS "Processing ${PROJECT_NAME}")


add_definitions(-DCL_SOURCE_DIR="${Main_SOURCE_DIR}/")
### Add headers and .cpp files ##########

sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)


#####################################################################
# Add projects
#####################################################################
add_subdirectory(hiddenlayer)
add_subdirectory(nnet)
add_subdirectory(cost)
add_subdirectory(cl_intf)


SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
sdk_setup_project_bin(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} OpenCL)
