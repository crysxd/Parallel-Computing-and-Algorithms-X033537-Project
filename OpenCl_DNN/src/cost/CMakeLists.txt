#####################################################################
# Cost Project
#####################################################################
project(Cost)
message(STATUS "Processing ${PROJECT_NAME}")


#####################################################################
# Include the project files
#####################################################################
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)


#####################################################################
# Set to build as an executable
#####################################################################
add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
sdk_setup_project_lib(${PROJECT_NAME})
# add_dependencies(${PROJECT_NAME} Core)
# target_link_libraries(${PROJECT_NAME} Core)
