#####################################################################
# Root SDK level
#####################################################################
project(SDK)
cmake_minimum_required(VERSION 2.6)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message(STATUS "Processing ${PROJECT_NAME}")


#####################################################################
# Util functions to make our lives easier
#####################################################################

# Function to list all header files in the current directory, recursing into sub-directories
# HEADER_FILES - To be filled with the found header files
function(sdk_list_header_files HEADER_FILES)
    file(GLOB_RECURSE HEADER_FILES_TMP "*.h" "*.hpp" "*.inl" "*.pch")
    set(HEADER_FILES ${HEADER_FILES_TMP} PARENT_SCOPE)
endfunction()

# Function to list all source files in the current directory, recursing into sub-directories
# SOURCE_FILES - To be filled with the found source files
function(sdk_list_source_files HEADER_FILES)
    file(GLOB_RECURSE SOURCE_FILES_TMP "*.c" "*.cpp")
    set(SOURCE_FILES ${SOURCE_FILES_TMP} PARENT_SCOPE)
endfunction()

# Function to setup some standard project items
# PROJECTNAME - The name of the project being setup
# TARGETDIR - The target directory for output files (relative to CMAKE_SOURCE_DIR)
function(sdk_setup_project_common PROJECTNAME TARGETDIR)
    # Set the Debug and Release names
    set_target_properties(${PROJECTNAME} PROPERTIES DEBUG_OUTPUT_NAME ${PROJECTNAME}_d RELEASE_OUTPUT_NAME ${PROJECTNAME})

    # Add a post-build step for MSVC to copy the output to the target directory
    if(MSVC)
        add_custom_command(TARGET ${PROJECTNAME} POST_BUILD COMMAND copy \"$(TargetPath)\" \"${CMAKE_SOURCE_DIR}/${TARGETDIR}\")
    endif()

    # Setup install to copy the built output to the target directory (for compilers that don't have post build steps)
    install(
        TARGETS ${PROJECTNAME}
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
        ARCHIVE DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
        )
endfunction()

# Function to setup some project items for an executable or DLL
# PROJECTNAME - The name of the project being setup
function(sdk_setup_project_bin PROJECTNAME)
    sdk_setup_project_common(${PROJECTNAME} Bin)
endfunction()

# Function to setup some project items for static library
# PROJECTNAME - The name of the project being setup
function(sdk_setup_project_lib PROJECTNAME)
    sdk_setup_project_common(${PROJECTNAME} Lib)
endfunction()


#####################################################################
# Set up the basic build environment
#####################################################################
if(CMAKE_BUILD_TYPE STREQUAL "")
    # CMake defaults to leaving CMAKE_BUILD_TYPE empty. This screws up differentiation between debug and release builds.
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Set compiler specific build flags
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-msse)
endif()

if(MSVC)
    if(CMAKE_BUILD_TOOL STREQUAL "nmake")
        # Set variable to state that we are using nmake makefiles
        set(NMAKE TRUE)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
    # Enable intrinsics on MSVC in debug mode
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Oi")
    if(CMAKE_CL_64)
        # Visual Studio bails out on debug builds in 64bit mode unless this flag is set
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /bigobj")
    endif()
endif()

if(MINGW)
    add_definitions(-D_WIN32_WINNT=0x0500)
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)
    # Test for GCC visibility
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-fvisibility=hidden SDK_GCC_VISIBILITY)
    if(SDK_GCC_VISIBILITY)
        # Determine gcc version
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE SDK_GCC_VERSION)
        message(STATUS "Detected g++ ${SDK_GCC_VERSION}")
        message(STATUS "Enabling GCC visibility flags")
        set(SDK_GCC_VISIBILITY_FLAGS "-DSDK_GCC_VISIBILITY -fvisibility=hidden")

        # Check if we can safely add -fvisibility-inlines-hidden
        string(TOLOWER "${CMAKE_BUILD_TYPE}" SDK_BUILD_TYPE)
        if(SDK_BUILD_TYPE STREQUAL "debug" AND SDK_GCC_VERSION VERSION_LESS "4.2")
            message(STATUS "Skipping -fvisibility-inlines-hidden due to possible bug in g++ < 4.2")
        else()
            set(SDK_GCC_VISIBILITY_FLAGS "${SDK_GCC_VISIBILITY_FLAGS} -fvisibility-inlines-hidden")
        endif()
    endif()

    # Fix x64 issues on Linux
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64" AND NOT APPLE)
        add_definitions(-fPIC)
    endif()
endif()


#####################################################################
# Add the SDK projects
#####################################################################
include_directories("${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src)

add_executable(opencl src/main.cpp)
