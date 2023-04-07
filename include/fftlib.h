/*
  ==============================================================================

   FFT

  ==============================================================================
*/

#pragma once

#ifdef __SSE__
#include <xmmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fftlib {

// SIMD
#if defined(__SSE__)
#define SIMD_SUPPORT
typedef __m128 float4;
#define FLOAT4_LOAD(p) _mm_load_ps(p)
#define FLOAT4_STORE(p, v) _mm_store_ps(p, v)
#define FLOAT4_ADD(a, b) _mm_add_ps(a, b)
#define FLOAT4_SUB(a, b) _mm_sub_ps(a, b)
#define FLOAT4_MUL(a, b) _mm_mul_ps(a, b)
// Split interleaved complex array to real array and imag array
// [re0,im0,re1,im1,re2,im2,re3,im3] ->
//  ^ src_lo         ^ src_hi
// [re0,re1,re2,re3,im0,im1,im2,im3]
//  ^ dst_re         ^ dst_im
#define FLOAT4_SPLIT_REAL_IMAG(dst_re, dst_im, src_lo, src_hi)    \
  dst_re = _mm_shuffle_ps(x0_lo, x0_hi, _MM_SHUFFLE(2, 0, 2, 0)); \
  dst_im = _mm_shuffle_ps(x0_lo, x0_hi, _MM_SHUFFLE(3, 1, 3, 1));
// Interleave (redo splitting) real and imag
// [re0,re1,re2,re3,im0,im1,im2,im3] ->
//  ^ y0_re         ^ y0_im
// [re0,im0,re1,im1,re2,im2,re3,im3]
//  ^ y0_lo         ^ y0_hi
#define FLOAT4_INTERLEAVE_REAL_IMAG(dst_lo, dst_hi, src_re, src_im) \
  dst_lo = _mm_unpacklo_ps(src_re, src_im);                         \
  dst_hi = _mm_unpackhi_ps(src_re, src_im);
#elif defined(__ARM_NEON)
#define SIMD_SUPPORT
typedef float32x4_t float4;
#define FLOAT4_LOAD(p) vld1q_f32(p)
#define FLOAT4_STORE(p, v) vst1q_f32(p, v)
#define FLOAT4_ADD(a, b) vaddq_f32(a, b)
#define FLOAT4_SUB(a, b) vsubq_f32(a, b)
#define FLOAT4_MUL(a, b) vmulq_f32(a, b)
// Split interleaved complex array to real array and imag array
// [re0,im0,re1,im1,re2,im2,re3,im3] ->
//  ^ src_lo         ^ src_hi
// [re0,re1,re2,re3,im0,im1,im2,im3]
//  ^ dst_re         ^ dst_im
#define FLOAT4_SPLIT_REAL_IMAG(dst_re, dst_im, src_lo, src_hi) \
  {                                                            \
    float32x4x2_t unziped = vuzpq_f32(src_lo, src_hi);         \
    dst_re = unziped.val[0];                                   \
    dst_im = unziped.val[1];                                   \
  }
// Interleave (redo splitting) real and imag
// [re0,re1,re2,re3,im0,im1,im2,im3] ->
//  ^ y0_re         ^ y0_im
// [re0,im0,re1,im1,re2,im2,re3,im3]
//  ^ y0_lo         ^ y0_hi
#define FLOAT4_INTERLEAVE_REAL_IMAG(dst_lo, dst_hi, src_re, src_im) \
  {                                                                 \
    float32x4x2_t ziped = vzipq_f32(src_re, src_im);                \
    dst_lo = ziped.val[0];                                          \
    dst_hi = ziped.val[1];                                          \
  }
#endif

bool isPowerOfTwo(int x) { return (x & (x - 1)) == 0; }

unsigned long bitLength(unsigned long x) {
  unsigned long length = 0;
  while (x) {
    x >>= 1;
    length++;
  }
  return length;
}

unsigned long reverseBits(unsigned long x, unsigned long bit_length) {
  unsigned long rev = 0;
  for (unsigned long i = 0; i < bit_length; i++) {
    rev <<= 1;
    if ((x & 1) == 1) rev += 1;
    x >>= 1;
  }
  return rev;
}

template <class T>
std::complex<T> twiddleFactor(int x, size_t n) {
  constexpr T pi = static_cast<T>(M_PI);
  std::complex<T> w_angle = {
      static_cast<T>(0.0),
      static_cast<T>(-1.0 * 2.0) * pi / static_cast<T>(n) * static_cast<T>(x)};
  return std::exp(w_angle);
}

// Only useful for debug
template <class T>
[[maybe_unused]] void printVec(T* vec, size_t n) {
  for (size_t i = 0; i < n; i++) {
    std::cout << std::fixed << std::setw(7) << std::setprecision(2)
              << std::right;
    std::cout << vec[i] << (i + 1 < n ? "," : "\n");
  }
}

// FFT implementation with Cooley–Tukey FFT algorithm (decimation in time)
template <class T>
class Fft {
 public:
  Fft(size_t order)
      : order(order),
        n(static_cast<size_t>(std::pow(2, order))),
        index_bit_len(bitLength(n - 1)),
        w_arr(n / 2),
        w_arr_inverse(n / 2),
        bit_reverse_arr(n) {
    constexpr T pi = static_cast<T>(M_PI);

    for (size_t i = 0; i < n / 2; i++) {
      // Calculate twiddle factor ('W' in textbooks)
      std::complex<T> w_angle = {static_cast<T>(0.0),
                                 static_cast<T>(-1.0 * 2.0) * pi /
                                     static_cast<T>(n) * static_cast<T>(i)};
      w_arr[i] = std::exp(w_angle);
      w_arr_inverse[i] = std::exp(w_angle * static_cast<T>(-1.0));
    }

    for (size_t i = 0; i < n; i++) {
      bit_reverse_arr[i] = reverseBits(i, index_bit_len);
    }
  }

  // Returns if success
  bool fft(const std::complex<T>* input_buf, std::complex<T>* output_buf,
           bool inverse = false) {
    constexpr bool SUCCESS = true;

    // Copy input and reorder with bit inversion at the same time
    for (size_t i = 0; i < n; i++) {
      auto j = reverseBitsCached(i);
      output_buf[i] = input_buf[j];
    }

    if (n == 1) {
      return SUCCESS;
    }

    const auto num_loop = index_bit_len;
    const auto& _w_arr = inverse ? w_arr_inverse : w_arr;

    // Calculate FFT using butterfly operation
    for (size_t i = 0; i < num_loop; i++) {
      const auto num_element_per_group =
          static_cast<size_t>(std::pow(2, i + 1));
      ;
      const auto num_group = n / num_element_per_group;
      for (size_t j = 0; j < num_group; j++) {
        // k0: index for the first half of group
        // k1: index for the second half of group
        const auto k0_start = j * num_element_per_group;
        const auto k0_end =
            j * num_element_per_group + num_element_per_group / 2;  // exclusive
        for (size_t k0 = k0_start; k0 < k0_end; k0++) {
          const auto k1 = k0 + num_element_per_group / 2;
          const auto idx = k0 - k0_start;
          const auto x0 = output_buf[k0];
          const auto x1 = output_buf[k1];
          const std::complex<T> w = _w_arr[idx * num_group];
          const auto x0_re = x0.real();
          const auto x0_im = x0.imag();
          const auto x1_re = x1.real();
          const auto x1_im = x1.imag();
          const auto w_re = w.real();
          const auto w_im = w.imag();
          const auto wx1_re = w_re * x1_re - w_im * x1_im;
          const auto wx1_im = w_im * x1_re + w_re * x1_im;
          const auto y0_re = x0_re + wx1_re;
          const auto y0_im = x0_im + wx1_im;
          const auto y1_re = x0_re - wx1_re;
          const auto y1_im = x0_im - wx1_im;
          output_buf[k0] = {y0_re, y0_im};  // x0 + w * x1
          output_buf[k1] = {y1_re, y1_im};  // x0 - w * x1
        }
      }
    }

    // 'N' in textbook
    T divisor = inverse ? static_cast<T>(n) : 1.0;
    for (size_t i = 0; i < n; i++) {
      output_buf[i] = output_buf[i] / divisor;
    }

    return SUCCESS;
  }

  // Returns if success
  bool fft(const std::vector<std::complex<T>>& input_buf,
           std::vector<std::complex<T>>& output_buf, bool inverse = false) {
    constexpr bool SUCCESS = true;
    if (n != input_buf.size() || n != output_buf.size()) {
      return !SUCCESS;
    }
    return fft(input_buf.data(), output_buf.data(), inverse);
  }

  std::vector<std::complex<T>> fft(
      const std::vector<std::complex<T>>& input_buf, bool inverse = false) {
    auto n = input_buf.size();
    std::vector<std::complex<T>> output_buf(n);
    if (fft(input_buf, output_buf, inverse)) {
      return output_buf;
    } else {
      return {};
    }
  }

 private:
  const size_t order;
  const size_t n;
  const size_t index_bit_len;
  // pre-calculated caches
  // Twiddle factor ('W' in textbooks)
  std::vector<std::complex<T>> w_arr;
  std::vector<std::complex<T>> w_arr_inverse;
  std::vector<unsigned long> bit_reverse_arr;

  inline unsigned long reverseBitsCached(unsigned long x) {
    assert(x < bit_reverse_arr.size());
    return bit_reverse_arr[x];
  }
};

#ifdef SIMD_SUPPORT
// FFfloat implementation with Cooley–floatukey FFfloat algorithm (decimation in
// time)
template <>
class Fft<float> {
 public:
  Fft(size_t order)
      : n(static_cast<size_t>(std::pow(2, order))),
        index_bit_len(bitLength(n - 1)),
        w_arr(n / 2),
        w_arr_inverse(n / 2),
        w_arr_split(),
        w_arr_inverse_split(),
        bit_reverse_arr(n) {
    constexpr float pi = static_cast<float>(M_PI);

    for (size_t i = 0; i < n / 2; i++) {
      // Calculate twiddle factor ('W' in textbooks)
      std::complex<float> w_angle = {static_cast<float>(0.0),
                                     static_cast<float>(-1.0 * 2.0) * pi /
                                         static_cast<float>(n) *
                                         static_cast<float>(i)};
      w_arr[i] = std::exp(w_angle);
      w_arr_inverse[i] = std::exp(w_angle * static_cast<float>(-1.0));
    }

    const auto num_loop = index_bit_len;
    for (size_t i = 0; i < num_loop; i++) {
      const auto num_element_per_group =
          static_cast<size_t>(std::pow(2, i + 1));
      ;
      const auto num_group = n / num_element_per_group;
      const int num_paralell = 4;
      w_arr_split.emplace_back();
      w_arr_inverse_split.emplace_back();
      for (size_t idx = 0; idx < num_element_per_group / 2;
           idx += num_paralell) {
        w_arr_split.back().emplace_back();
        auto& w_slot = w_arr_split.back().back();
        w_slot[0] = twiddleFactor<float>((idx + 0) * num_group, n).real();
        w_slot[1] = twiddleFactor<float>((idx + 1) * num_group, n).real();
        w_slot[2] = twiddleFactor<float>((idx + 2) * num_group, n).real();
        w_slot[3] = twiddleFactor<float>((idx + 3) * num_group, n).real();
        w_slot[4] = twiddleFactor<float>((idx + 0) * num_group, n).imag();
        w_slot[5] = twiddleFactor<float>((idx + 1) * num_group, n).imag();
        w_slot[6] = twiddleFactor<float>((idx + 2) * num_group, n).imag();
        w_slot[7] = twiddleFactor<float>((idx + 3) * num_group, n).imag();
        w_arr_inverse_split.back().emplace_back();
        auto& wi_slot = w_arr_inverse_split.back().back();
        wi_slot[0] = twiddleFactor<float>(-1 * (idx + 0) * num_group, n).real();
        wi_slot[1] = twiddleFactor<float>(-1 * (idx + 1) * num_group, n).real();
        wi_slot[2] = twiddleFactor<float>(-1 * (idx + 2) * num_group, n).real();
        wi_slot[3] = twiddleFactor<float>(-1 * (idx + 3) * num_group, n).real();
        wi_slot[4] = twiddleFactor<float>(-1 * (idx + 0) * num_group, n).imag();
        wi_slot[5] = twiddleFactor<float>(-1 * (idx + 1) * num_group, n).imag();
        wi_slot[6] = twiddleFactor<float>(-1 * (idx + 2) * num_group, n).imag();
        wi_slot[7] = twiddleFactor<float>(-1 * (idx + 3) * num_group, n).imag();
      }
    }

    for (size_t i = 0; i < n; i++) {
      bit_reverse_arr[i] = reverseBits(i, index_bit_len);
    }
  }

  // Returns if success
  bool fft(const std::complex<float>* input_buf,
           std::complex<float>* output_buf, bool inverse = false) {
    constexpr bool SUCCESS = true;

    // Copy input and reorder with bit inversion at the same time
    for (size_t i = 0; i < n; i++) {
      auto j = reverseBitsCached(i);
      output_buf[i] = input_buf[j];
    }

    if (n == 1) {
      return SUCCESS;
    }

    const auto num_loop = index_bit_len;
    const auto& _w_arr = inverse ? w_arr_inverse : w_arr;
    auto& _w_arr_split = inverse ? w_arr_inverse_split : w_arr_split;

    // Calculate FFfloat using butterfly operation
    for (size_t i = 0; i < num_loop; i++) {
      const auto num_element_per_group =
          static_cast<size_t>(std::pow(2, i + 1));
      ;
      const auto num_group = n / num_element_per_group;
      if (num_element_per_group <= 4) {
        // Do FFT normally
        for (size_t j = 0; j < num_group; j++) {
          // k0: index for the first half of group
          // k1: index for the second half of group
          const auto k0_start = j * num_element_per_group;
          const auto k0_end = j * num_element_per_group +
                              num_element_per_group / 2;  // exclusive
          for (size_t k0 = k0_start; k0 < k0_end; k0++) {
            const auto k1 = k0 + num_element_per_group / 2;
            const auto idx = k0 - k0_start;
            const auto x0 = output_buf[k0];
            const auto x1 = output_buf[k1];
            const std::complex<float> w = _w_arr[idx * num_group];
            const auto x0_re = x0.real();
            const auto x0_im = x0.imag();
            const auto x1_re = x1.real();
            const auto x1_im = x1.imag();
            const auto w_re = w.real();
            const auto w_im = w.imag();
            const auto wx1_re = w_re * x1_re - w_im * x1_im;
            const auto wx1_im = w_im * x1_re + w_re * x1_im;
            const auto y0_re = x0_re + wx1_re;
            const auto y0_im = x0_im + wx1_im;
            const auto y1_re = x0_re - wx1_re;
            const auto y1_im = x0_im - wx1_im;
            output_buf[k0] = {y0_re, y0_im};  // x0 + w * x1
            output_buf[k1] = {y1_re, y1_im};  // x0 - w * x1
          }
        }
      } else {
        // Do SIMD FFT
        for (size_t j = 0; j < num_group; j++) {
          // k0: index for the first half of group
          // k1: index for the second half of group
          const auto k0_start = j * num_element_per_group;
          const auto k0_end = j * num_element_per_group +
                              num_element_per_group / 2;  // exclusive
          for (size_t k0 = k0_start; k0 < k0_end; k0 += 4) {
            const auto k1 = k0 + num_element_per_group / 2;
            const auto idx = k0 - k0_start;
            float4 x0_re;
            float4 x0_im;
            float4 x1_re;
            float4 x1_im;
            if (num_element_per_group == 8) {
              // Split interleaved complex array to real array and imag array
              // [re0,im0,re1,im1,re2,im2,re3,im3] ->
              //  ^ x0_lo         ^ x0_hi
              // [re0,re1,re2,re3,im0,im1,im2,im3]
              //  ^ x0_re         ^ x0_im
              float4 x0_lo =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k0]));
              float4 x0_hi =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k0 + 2]));
              FLOAT4_SPLIT_REAL_IMAG(x0_re, x0_im, x0_lo, x0_hi)
              float4 x1_lo =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k1]));
              float4 x1_hi =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k1 + 2]));
              FLOAT4_SPLIT_REAL_IMAG(x1_re, x1_im, x1_lo, x1_hi)
            } else {
              x0_re = FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k0]));
              x0_im =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k0 + 2]));
              x1_re = FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k1]));
              x1_im =
                  FLOAT4_LOAD(reinterpret_cast<float*>(&output_buf[k1 + 2]));
            }
            constexpr int num_parallel = 4;
            const int iteration_count = idx / num_parallel;
            float4 w_re = FLOAT4_LOAD(
                reinterpret_cast<float*>(&_w_arr_split[i][iteration_count]));
            float4 w_im = FLOAT4_LOAD(
                reinterpret_cast<float*>(&_w_arr_split[i][iteration_count]) +
                4 /*offset to imag*/);
            float4 wx1_re = FLOAT4_SUB(
                FLOAT4_MUL(w_re, x1_re),
                FLOAT4_MUL(w_im, x1_im));  // w_re * x1_re - w_im * x1_im
            float4 wx1_im = FLOAT4_ADD(
                FLOAT4_MUL(w_im, x1_re),
                FLOAT4_MUL(w_re, x1_im));  // w_im * x1_re + w_re * x1_im
            float4 y0_re = FLOAT4_ADD(x0_re, wx1_re);  // x0_re + wx1_re
            float4 y0_im = FLOAT4_ADD(x0_im, wx1_im);  // x0_im + wx1_im
            float4 y1_re = FLOAT4_SUB(x0_re, wx1_re);  // x0_re - wx1_re
            float4 y1_im = FLOAT4_SUB(x0_im, wx1_im);  // x0_im - wx1_im
            if (i + 1 == num_loop && num_element_per_group >= 8) {
              // Redo splitting real and imag
              // [re0,re1,re2,re3,im0,im1,im2,im3] ->
              //  ^ y0_re         ^ y0_im
              // [re0,im0,re1,im1,re2,im2,re3,im3]
              //  ^ y0_lo         ^ y0_hi
              float4 y0_lo;
              float4 y0_hi;
              float4 y1_lo;
              float4 y1_hi;
              FLOAT4_INTERLEAVE_REAL_IMAG(y0_lo, y0_hi, y0_re, y0_im)
              FLOAT4_INTERLEAVE_REAL_IMAG(y1_lo, y1_hi, y1_re, y1_im)
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k0]), y0_lo);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k0 + 2]),
                           y0_hi);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k1]), y1_lo);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k1 + 2]),
                           y1_hi);
            } else {
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k0]), y0_re);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k0 + 2]),
                           y0_im);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k1]), y1_re);
              FLOAT4_STORE(reinterpret_cast<float*>(&output_buf[k1 + 2]),
                           y1_im);
            }
          }
        }
      }
    }

    // 'N' in textbook
    float divisor = inverse ? static_cast<float>(n) : 1.0;
    for (size_t i = 0; i < n; i++) {
      output_buf[i] = output_buf[i] / divisor;
    }

    return SUCCESS;
  }

  // Returns if success
  bool fft(const std::vector<std::complex<float>>& input_buf,
           std::vector<std::complex<float>>& output_buf, bool inverse = false) {
    constexpr bool SUCCESS = true;
    if (n != input_buf.size() || n != output_buf.size()) {
      return !SUCCESS;
    }
    return fft(input_buf.data(), output_buf.data(), inverse);
  }

  std::vector<std::complex<float>> fft(
      const std::vector<std::complex<float>>& input_buf, bool inverse = false) {
    auto n = input_buf.size();
    std::vector<std::complex<float>> output_buf(n);
    if (fft(input_buf, output_buf, inverse)) {
      return output_buf;
    } else {
      return {};
    }
  }

 private:
  const size_t n;
  const size_t index_bit_len;
  // pre-calculated caches
  // floatwiddle factor ('W' in textbooks)
  std::vector<std::complex<float>> w_arr;
  std::vector<std::complex<float>> w_arr_inverse;
  std::vector<std::vector<std::array<float, 8 /*for 4 float complex number*/>>>
      w_arr_split;
  std::vector<std::vector<std::array<float, 8>>> w_arr_inverse_split;
  std::vector<unsigned long> bit_reverse_arr;

  inline unsigned long reverseBitsCached(unsigned long x) {
    assert(x < bit_reverse_arr.size());
    return bit_reverse_arr[x];
  }
};
#endif

}  // namespace fftlib
