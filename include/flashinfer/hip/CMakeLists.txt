
cmake_minimum_required(VERSION 3.18)

project(FlashInfer VERSION 1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "/opt/rocm/cmake/hip/") 
find_package(HIP REQUIRED)

include_directories(
    "/root/flashinfer/include/flashinfer/hip"  
    ${HIP_INCLUDE_DIRS}
)

set(SOURCES
    main.cpp 
)

add_executable(flashinfer ${SOURCES})

target_link_libraries(flashinfer
    ${HIP_LIBRARIES}
)


set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -g -D__HIP_PLATFORM_HCC__")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set_target_properties(flashinfer PROPERTIES
    HIP_SEPARABLE_COMPILATION ON
    HIP_ARCHITECTURES "gfx942"
)

set(CMAKE_BUILD_TYPE Release)  

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
