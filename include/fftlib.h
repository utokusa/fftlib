/*
  ==============================================================================

   FFT

  ==============================================================================
*/

#pragma once

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace fftlib {

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
class Fft {
 public:
  Fft(size_t order)
      : order(order),
        n(static_cast<size_t>(std::pow(2, order))),
        index_bit_len(bitLength(n - 1)),
        w_arr(n),
        w_arr_inverse(n),
        bit_reverse_arr(n / 2) {
    constexpr T pi = static_cast<T>(M_PI);
    for (size_t i = 0; i < n; i++) {
      std::complex<T> w_angle = {static_cast<T>(0.0),
                                 static_cast<T>(-1.0 * 2.0) * pi /
                                     static_cast<T>(n) * static_cast<T>(i)};
      w_arr[i] = std::exp(w_angle);
      w_arr_inverse[i] = std::exp(w_angle * static_cast<T>(-1.0));

      for (size_t i = 0; i < n / 2; i++) {
        bit_reverse_arr[i] = reverseBits(i, index_bit_len);
      }
    }
  }

  // Returns if success
  bool fft(const std::complex<T>* input_buf, std::complex<T>* output_buf,
           bool inverse = false) {
    constexpr bool SUCCESS = true;
    // Copy input
    for (size_t i = 0; i < n; i++) {
      output_buf[i] = input_buf[i];
    }

    if (n == 1) {
      return SUCCESS;
    }

    const auto num_loop = index_bit_len;

    // Calculate FFT using butterfly operation
    for (size_t i = 0; i < num_loop; i++) {
      const auto num_group = static_cast<size_t>(std::pow(2, i));
      const auto num_element_per_group = n / num_group;
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
          const std::complex<T> w = inverse ? wInverseCached(idx, num_group)
                                            : wCached(idx, num_group);
          output_buf[k0] = x0 + x1;
          output_buf[k1] = w * (x0 - x1);
        }
      }
    }

    // 'N' in textbook
    T divisor = inverse ? static_cast<T>(n) : 1.0;
    for (size_t i = 0; i < n; i++) {
      output_buf[i] = output_buf[i] / divisor;
    }

    // Restore order of output using bit inversion
    for (size_t i = 0; i < n / 2; i++) {
      auto j = reverseBitsCached(i);
      swap(output_buf[i], output_buf[j]);
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
  std::vector<std::complex<T>> w_arr;
  std::vector<std::complex<T>> w_arr_inverse;
  std::vector<unsigned long> bit_reverse_arr;

  inline std::complex<T> wCached(size_t idx, size_t num_group) {
    const size_t w_arr_idx = idx * num_group;
    assert(w_arr_idx < w_arr.size());
    return w_arr[w_arr_idx];
  }

  inline std::complex<T> wInverseCached(size_t idx, size_t num_group) {
    const size_t w_arr_idx = idx * num_group;
    assert(w_arr_idx < w_arr_inverse.size());
    return w_arr_inverse[w_arr_idx];
  }

  inline unsigned long reverseBitsCached(unsigned long x) {
    assert(x < bit_reverse_arr.size());
    return bit_reverse_arr[x];
  }
};

}  // namespace fftlib
