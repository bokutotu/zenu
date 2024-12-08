cmake_minimum_required(VERSION 3.18)

project(cudnn_frontend_wrapper VERSION 1.8.0)

option(CUDNN_FRONTEND_SKIP_JSON_LIB "Defines whether FE should not include nlohmann/json.hpp." OFF)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)

add_library(cudnn_frontend INTERFACE)

# Set include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set compile definitions
add_compile_definitions(
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)

# Add source files
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add static library
add_library(cudnn_frontend_wrapper STATIC ${SRC_FILES})

target_compile_options(cudnn_frontend_wrapper PUBLIC
    -I${CMAKE_SOURCE_DIR}/cudnn-frontend/include
)

# Link libraries
target_link_libraries(cudnn_frontend_wrapper
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    CUDNN::cudnn
)

# Set include directories for the target
target_include_directories(cudnn_frontend_wrapper PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Add debug information for Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cudnn_frontend_wrapper PUBLIC -g)
endif()

if(CUDNN_FRONTEND_BUILD_TESTS)
    # Add Catch2
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7
    )
    FetchContent_MakeAvailable(Catch2)

    # Add test source files
    file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp)

    # Add test executable
    add_executable(run_tests ${TEST_SRC_FILES})

    # Link libraries to test executable
    target_link_libraries(run_tests
        cudnn_frontend_wrapper
        Catch2::Catch2
    )

    # Add test to CTest
    include(CTest)
    add_test(NAME run_tests COMMAND run_tests)
endif()

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/.ccls "
%compile_commands.json=build/compile_commands.json
%clang
-std=c++17
-I${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
")
