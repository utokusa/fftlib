/*
  ==============================================================================

   FFT

  ==============================================================================
*/

#pragma once

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

std::vector<std::complex<double>> fft(
    const std::vector<std::complex<double>>& input_buf, bool inverse = false) {
  auto n = input_buf.size();
  if (n == 0 || !isPowerOfTwo(static_cast<int>(n))) {
    return {};
  }
  if (n == 1) {
    return input_buf;
  }

  std::vector<std::complex<double>> output_buf(n);
  // Copy input
  for (size_t i = 0; i < n; i++) {
    output_buf[i] = input_buf[i];
  }
  auto index_bit_len = bitLength(n - 1);
  auto num_loop = index_bit_len;

  // Calculate FFT using butterfly operation
  for (size_t i = 0; i < num_loop; i++) {
    auto num_group = static_cast<size_t>(std::pow(2, i));
    auto num_element_per_group = n / num_group;
    for (size_t j = 0; j < num_group; j++) {
      // k0: index for the first half of group
      // k1: index for the second half of group
      auto k0_start = j * num_element_per_group;
      auto k0_end =
          j * num_element_per_group + num_element_per_group / 2;  // exclusive
      for (size_t k0 = k0_start; k0 < k0_end; k0++) {
        auto k1 = k0 + num_element_per_group / 2;
        auto idx = k0 - k0_start;
        auto x0 = output_buf[k0];
        auto x1 = output_buf[k1];
        auto angle_sign = inverse ? -1.0 : 1.0;
        constexpr double pi = M_PI;
        std::complex<double> w_angle = {
            0.0, -1.0 * 2.0 * pi / static_cast<double>(num_element_per_group) *
                     static_cast<double>(idx) * angle_sign};
        std::complex<double> w = std::exp(w_angle);
        output_buf[k0] = x0 + x1;
        output_buf[k1] = w * (x0 - x1);
      }
    }
  }

  // 'N' in textbook
  double divisor = inverse ? static_cast<double>(n) : 1.0;
  for (size_t i = 0; i < n; i++) {
    output_buf[i] = output_buf[i] / divisor;
  }

  // Restore order of output using bit inversion
  for (size_t i = 0; i < n / 2; i++) {
    auto j = reverseBits(i, index_bit_len);
    swap(output_buf[i], output_buf[j]);
  }

  return output_buf;
}
}  // namespace fftlib
