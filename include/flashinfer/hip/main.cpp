#include "math.hip.h"

#include <iostream>
#include <cstring>

__global__ void __test_ptx_exp2_kernel(float* d_input_, float* d_output_, int size_) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size_) {
        d_output_[idx] = flashinfer::math::ptx_exp2(d_input_[idx]);
    }
}

__global__ void __test_ptx_rcp_kernel(float* d_input_, float* d_output_, int size_) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size_) {
        d_output_[idx] = flashinfer::math::ptx_rcp(d_input_[idx]);
    }
}

class TestObject{
    public:
    TestObject(size_t size) : size_(size){
        BYTES_ = size_ * sizeof(float);
        h_input_ = new float[size_];
        h_output_ = new float[size_];

        hipMalloc((void**)&d_input_, BYTES_);
        hipMalloc((void**)&d_output_, BYTES_);

        grid_size_ = (size_ + block_size_ - 1) / block_size_;  

        __reset();
    }


    void launch_ptx_exp2_kernel(){

        __test_ptx_exp2_kernel<<<block_size_, grid_size_>>>(d_input_, d_output_, size_);

        hipMemcpy(h_output_, d_output_, BYTES_, hipMemcpyDeviceToHost);

        std::cout << "Results of ptx_exp2 function:" << std::endl;
        int count{0};
        for (int i = 0; i < size_; ++i) {
            if(std::pow(2.0f, h_input_[i]) != h_output_[i]){
                std::cout<<"Mismatch at "<<i<<std::endl;
            }
            else{
                ++count;
            }
        }

        if(count == size_){
            std::cout<<"PTX_EXP TEST Passed Successfully \n!!"<<std::endl;
        }

        __reset();
    }

    void launch_ptx_rcp_kernel(){

        __test_ptx_rcp_kernel<<<block_size_, grid_size_>>>(d_input_, d_output_, size_);

        hipMemcpy(h_output_, d_output_, BYTES_, hipMemcpyDeviceToHost);

        std::cout << "Results of ptx_rcp function:" << std::endl;
        int count{0};
        for (int i = 0; i < size_; ++i) {
            if(1.0f/h_input_[i] != h_output_[i]){
                std::cout<<"Mismatch at "<<i<<std::endl;
            }
            else{
                ++count;
            }
        }

        if(count == size_){
            std::cout<<"PTX_RCP TEST Passed Successfully!! \n"<<std::endl;
        }

        __reset();
    }

    ~TestObject(){
        delete[] h_input_;
        delete[] h_output_;
        hipFree(d_input_);
        hipFree(d_output_);
    }


    
    // __global__ void test_shfl_xor_sync(float* input, float* output, int lane_mask){
    //     int lane = threadIdx.x % 64;
    //     float val = input[lane];
    //     float result = __shfl_xor_sync(0xffffffffffffffff, val, lane_mask);
    //     output[lane] = result;
    // }

    private:
    void __reset(){

        for(size_t i = 0; i < size_; ++i){
            int var = static_cast<float>(rand() % 12);
            int sign = var % 2 == 0 ? 1 : -1;
            h_input_[i] = (static_cast<float>(var) / 13.89) * sign;
        }
        std::memset(h_output_, 0.0f, BYTES_);
        hipMemset(d_input_, 0, BYTES_);
        hipMemset(d_output_, 0, BYTES_);
        hipMemcpy(d_input_, h_input_, BYTES_, hipMemcpyHostToDevice);
    }



    private:
    size_t size_{};
    size_t BYTES_{};
    float* h_input_;
    float* h_output_;
    float* d_input_;
    float* d_output_;
    size_t block_size_{256};  // Threads per block
    size_t grid_size_{}; // Number of blocks
};

void launch_test_shfl_xor_sync(){
    const int WARP_SIZE = 64;
    float h_input[WARP_SIZE], h_output[WARP_SIZE];

    float *d_input, *d_output;
    int lane_mask = 1;

    for(int i = 0; i < WARP_SIZE; ++i){
        h_input[i] = static_cast<float>(i);
    }
    
    size_t BYTES = WARP_SIZE * sizeof(float);

    hipMalloc((void**)&d_input, BYTES);
    hipMalloc((void**)&d_output, BYTES);

    hipMemcpy(d_input, h_input, BYTES, hipMemcpyHostToDevice);

    test_shfl_xor_sync<<<1, WARP_SIZE>>>(d_input, d_output, lane_mask);
    hipMemcpy(h_output, d_output, BYTES, hipMemcpyDeviceToHost);

    for(int i = 0; i < WARP_SIZE; ++i){
        int expected_idx = i ^ lane_mask;
        if(expected_idx < WARP_SIZE){
            if(h_output[i] != h_input[expected_idx]){
                std::cout<<"Mismatch at lane "<<i<<std::endl;
            }
        }
    }

    hipFree(d_input);
    hipFree(d_output);
}
int main() {
    
    TestObject obj(5);
    obj.launch_ptx_exp2_kernel();
    obj.launch_ptx_rcp_kernel();
}