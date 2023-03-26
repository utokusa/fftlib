#include <fftlib.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <utility>

#include "benchmark_util.h"

namespace fftlib {

TEST_CASE("fftlib benchmark", "[!benchmark]") {
  BENCHMARK_ADVANCED("fft 8")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec(8);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    meter.measure([&]() { fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 32")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec(32);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    meter.measure([&]() { fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 64")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec(64);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    meter.measure([&]() { fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 512")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec(512);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    meter.measure([&]() { fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 1024")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec(1024);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    meter.measure([&]() { fft(input_buf, output_buf); });
  };
}
}  // namespace fftlib
