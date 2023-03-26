/*
  ==============================================================================

   Utility for benchmark

  ==============================================================================
*/

#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

namespace fftlib {
std::vector<std::complex<double>> genRandVec(size_t n, bool real_value = false,
                                             unsigned int seed = 0) {
  std::default_random_engine rand_engine(seed);
  std::uniform_real_distribution<> rand_dist(0.0, 1.0);
  std::vector<std::complex<double>> ret(n);
  for (size_t i = 0; i < n; i++) {
    ret[i] = {rand_dist(rand_engine),
              real_value ? 0.0 : rand_dist(rand_engine)};
  }

  return ret;
}
}  // namespace fftlib
