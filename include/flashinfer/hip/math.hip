#define HIP_ENABLE_WARP_SYNC_BUILTINS

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

template <typename T>
__forceinline__ __device__ T ptx_exp2(T x);

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct equivalent of __exp2f()
/// @param x Input power to exponentiate  
/// @return Computes 2 ^ x 
template<>
__forceinline__ __device__ float ptx_exp2<float>(float x) {
  return __exp10f(x * __log10f(2.0f)); // Writing 2^x = 10 ^ (x * log_10(2))
}

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct equivalent of __exp2f()
/// @param x Input power to exponentiate  
/// @return Computes 2 ^ x 
template<>
__forceinline__ __device__ half ptx_exp2<half>(half x) {
  return hexp2(x);
}

/// @brief Wrapper for computing 2 ^ x. We currently do not support a direct equivalent of __exp2f()
/// @param x Vector of two half dtypes to exponentiate  
/// @return Computes 2 ^ x 
template<>
__forceinline__ __device__ half2 ptx_exp2<half2>(half2 x) {
  return half2(ptx_exp2(x.x), ptx_exp2(x.y));
}

/// @brief Compute log2
/// @param x Input param - float dtype
/// @return Log2
__forceinline__ __device__ float ptx_log2(float x) {
  return __log2f(x);
}

/// @brief Compute 1/x
/// @param x Input param - float dtype
/// @return Returns 1 / x in round-to-nearest-even mod.
__forceinline__ __device__ float ptx_rcp(float x) {
  return 	__frcp_rn(x);
}

template <typename T>
__forceinline__ __device__ T shfl_xor_sync(T x, int lane_mask);

/// @brief Perform butterfly shuffle between threads in a warp
/// @param x The value in the source lane
/// @param lane_mask The mask to perform thread index xor with
template<>
__forceinline__ __device__ float shfl_xor_sync<float>(float x, int lane_mask) {
  return __shfl_xor_sync(0xffffffffffffffffULL, x, lane_mask);  // 0xffffffff ensures the full warp is considered
}

/// @brief Perform butterfly shuffle between threads in a warp
/// @param x Vector of two values in the source lane
/// @param lane_mask The mask to perform thread index xor with
template<>
__forceinline__ __device__ half2 shfl_xor_sync<half2>(half2 x, int lane_mask) {
  return __shfl_xor_sync(0xffffffffffffffffULL, x, lane_mask);  // 0xffffffff ensures the full warp is considered
}

/// @brief Wrapper for math intrinsic 1/sqrt(x)
/// @param x Input param - float dtype
/// @return Returns 1 / sqrt(x) in round to nearest even mode
__forceinline__ __device__ float rsqrt(float x) {
  return __frsqrt_rn(x);
}

template <typename T>
__forceinline__ __device__ T tanh(T x);

/// @brief Compute tanhf(x)
/// @param x Input param - float dtype
/// @return Returns tanhf(x)
/// @note ROCm6.3 does not have a fast tanh or instrincs to support this
template<>
__forceinline__ __device__ float tanh<float>(float x) {
  return tanhf(x);
}

/// @brief A utility function to compute tanh for half dtype
/// @param x Input param - half 
/// @return Hyperbolic tangent of x
template<>
__forceinline__ __device__ half tanh<half>(half x) {
  half a = hexp(x);
  half b = hexp(-x);

  return half((a-b)/(a+b));
}

/// @brief Compute hyperbolic tangent for a vector of two half dtype
/// @param x Vector of two half dtypes
/// @return Hyperbolic tangent of x
template<>
__forceinline__ __device__ half2 tanh<half2>(half2 x) {
  return half2(tanh(x.x), tanh(x.y));
}

}  // namespace math
}  // namespace flashinfer
