#include "math.hip"
#include <iostream>

__global__ void test_ptx_exp2_kernel(float* x_values, float* results, int num_values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_values) {
        results[idx] = flashinfer::math::ptx_exp2(x_values[idx]);
    }
}

int main() {
    const int num_values = 5;
    float x_host[num_values] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f}; // Input values for testing
    float results_host[num_values];  // Array to store the results from device

    // Device memory pointers
    float *x_device, *results_device;

    // Allocate device memory
    hipMalloc((void**)&x_device, num_values * sizeof(float));
    hipMalloc((void**)&results_device, num_values * sizeof(float));

    // Copy input data from host to device
    hipMemcpy(x_device, x_host, num_values * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid size
    int block_size = 256;  // Threads per block
    int grid_size = (num_values + block_size - 1) / block_size;  // Number of blocks

    // Launch the kernel
    test_ptx_exp2_kernel<<<grid_size, block_size>>>(x_device, results_device, num_values);

    // Copy the results back from device to host
    hipMemcpy(results_host, results_device, num_values * sizeof(float), hipMemcpyHostToDevice);

    // Print the results
    std::cout << "Results of ptx_exp2 function:" << std::endl;
    for (int i = 0; i < num_values; ++i) {
        std::cout << "2^" << x_host[i] << " = " << results_host[i] << std::endl;
    }

    // Free device memory
    hipFree(x_device);
    hipFree(results_device);

    return 0;
}