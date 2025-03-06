cmake_minimum_required(VERSION 3.21)

project(FlashInferMath LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HIP REQUIRED)
set(CMAKE_HIP_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --offload-arch=gfx942")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_library(math STATIC /root/flashinfer/include/flashinfer/hip/math.hip.h)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -g -D__HIP_PLATFORM_AMD__")

set_target_properties(math PROPERTIES
    HIP_SOURCES_PROPERTY_FORMAT 1
    HIP_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE HIP
)

target_link_libraries(math PUBLIC hip::host hip::device)

add_executable(test_math /root/flashinfer/include/flashinfer/hip/tests/test_math.cpp)

target_link_libraries(test_math PRIVATE
math
"${TORCH_LIBRARIES}"
${GTEST_LIBRARIES}
${GTEST_MAIN_LIBRARIES}
pthread
)

if(HIP_FOUND)
    set_target_properties(test_math PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

enable_testing()
add_test(NAME MathTest COMMAND test_math)
