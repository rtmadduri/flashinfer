#include "/root/flashinfer/include/flashinfer/hip/math.hip.h"

#include <gtest/gtest.h>
#include <torch/torch.h>


using namespace flashinfer::math;

constexpr int NUM_VALUES = 5;
constexpr size_t BLOCK_SIZE = 256;

template<typename T>
__global__ void test_ptx_exp2_kernel(T* x_values, T* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_VALUES) {
        results[idx] = ptx_exp2(x_values[idx]);
    }
}

__global__ void test_ptx_log2_kernel(float* x_values, float* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_VALUES) {
        results[idx] = ptx_log2(x_values[idx]);
    }
}

__global__ void test_ptx_rcp_kernel(float* x_values, float* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_VALUES) {
        results[idx] = ptx_rcp(x_values[idx]);
    }
}

__global__ void test_rsqrt_kernel(float* x_values, float* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_VALUES) {
        results[idx] = rsqrt(x_values[idx]);
    }
}

template<typename T>
__global__ void test_tanh_kernel(T* x_values, T* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_VALUES) {
        results[idx] = tanh(x_values[idx]);
    }
}

TEST(hipFunctionsTest, TestPtxExp2Float) {

    float x_host[NUM_VALUES] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    float results_host[NUM_VALUES]; 

    float *x_device, *results_device;

    hipMalloc((void**)&x_device, NUM_VALUES * sizeof(float));
    hipMalloc((void**)&results_device, NUM_VALUES * sizeof(float));

    hipMemcpy(x_device, x_host, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);


    int grid_size = (NUM_VALUES + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    test_ptx_exp2_kernel<<<grid_size, BLOCK_SIZE>>>(x_device, results_device);

    hipMemcpy(results_host, results_device, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < NUM_VALUES; ++i){
        x_host[i] = std::pow(2, x_host[i]);
    }

    for (int i = 0; i < NUM_VALUES; ++i) {
        EXPECT_NEAR(x_host[i], results_host[i], 1e-5);
    }

    hipFree(x_device);
    hipFree(results_device);
}

TEST(hipFunctionsTest, TestPtxLog2) {
    float x_host[NUM_VALUES] = {100.8, 37.85, 8.12f, 15.63, 29.0f};
    float results_host[NUM_VALUES]; 

    float *x_device, *results_device;

    hipMalloc((void**)&x_device, NUM_VALUES * sizeof(float));
    hipMalloc((void**)&results_device, NUM_VALUES * sizeof(float));

    hipMemcpy(x_device, x_host, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);


    int grid_size = (NUM_VALUES + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    test_ptx_log2_kernel<<<grid_size, BLOCK_SIZE>>>(x_device, results_device);

    hipMemcpy(results_host, results_device, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < NUM_VALUES; ++i){
        x_host[i] = std::log2f(x_host[i]);
    }

    for (int i = 0; i < NUM_VALUES; ++i) {
        EXPECT_NEAR(x_host[i], results_host[i], 1e-5);
    }

    hipFree(x_device);
    hipFree(results_device);
}

TEST(hipFunctionsTest, TestPtxRcp) {
    float x_host[NUM_VALUES] = {10.23f, 5.56f, 8.2f, 3.141f, 9.81f};
    float results_host[NUM_VALUES]; 

    float *x_device, *results_device;

    hipMalloc((void**)&x_device, NUM_VALUES * sizeof(float));
    hipMalloc((void**)&results_device, NUM_VALUES * sizeof(float));

    hipMemcpy(x_device, x_host, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);


    int grid_size = (NUM_VALUES + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    test_ptx_rcp_kernel<<<grid_size, BLOCK_SIZE>>>(x_device, results_device);

    hipMemcpy(results_host, results_device, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < NUM_VALUES; ++i){
        x_host[i] = 1.0f / x_host[i];
    }

    for (int i = 0; i < NUM_VALUES; ++i) {
        EXPECT_NEAR(x_host[i], results_host[i], 1e-5);
    }

    hipFree(x_device);
    hipFree(results_device);
}

TEST(hipFunctionsTest, TestRsqrt) {
    float x_host[NUM_VALUES] = {10.23f, 5.56f, 8.2f, 3.141f, 9.81f};
    float results_host[NUM_VALUES]; 

    float *x_device, *results_device;

    hipMalloc((void**)&x_device, NUM_VALUES * sizeof(float));
    hipMalloc((void**)&results_device, NUM_VALUES * sizeof(float));

    hipMemcpy(x_device, x_host, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);


    int grid_size = (NUM_VALUES + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    test_rsqrt_kernel<<<grid_size, BLOCK_SIZE>>>(x_device, results_device);

    hipMemcpy(results_host, results_device, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < NUM_VALUES; ++i){
        x_host[i] = 1.0f / std::sqrtf(x_host[i]);
    }

    for (int i = 0; i < NUM_VALUES; ++i) {
        EXPECT_NEAR(x_host[i], results_host[i], 1e-5);
    }

    hipFree(x_device);
    hipFree(results_device);
}

TEST(hipFunctionsTest, TestTanh) {
    float x_host[NUM_VALUES] = {3.5f, -2.2f, 1.5f, 1.83f, 0.87f};
    float results_host[NUM_VALUES]; 

    float *x_device, *results_device;

    hipMalloc((void**)&x_device, NUM_VALUES * sizeof(float));
    hipMalloc((void**)&results_device, NUM_VALUES * sizeof(float));

    hipMemcpy(x_device, x_host, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);


    int grid_size = (NUM_VALUES + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    test_tanh_kernel<<<grid_size, BLOCK_SIZE>>>(x_device, results_device);

    hipMemcpy(results_host, results_device, NUM_VALUES * sizeof(float), hipMemcpyHostToDevice);

    for(size_t i = 0; i < NUM_VALUES; ++i){
        x_host[i] = std::tanhf(x_host[i]);
    }

    for (int i = 0; i < NUM_VALUES; ++i) {
        EXPECT_NEAR(x_host[i], results_host[i], 1e-5);
    }

    hipFree(x_device);
    hipFree(results_device);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}