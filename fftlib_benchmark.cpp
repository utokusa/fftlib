#include <fftlib.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <utility>

#include "benchmark_util.h"

namespace fftlib {

TEST_CASE("fftlib benchmark (double)", "[!benchmark]") {
  BENCHMARK_ADVANCED("fft 8")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(8);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(3);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 32")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(32);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(5);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 64")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(64);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(6);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 512")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(512);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(9);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 1024")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(1024);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(10);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
  BENCHMARK_ADVANCED("fft 8192")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<double>> input_buf = genRandVec<double>(8192);
    std::vector<std::complex<double>> output_buf(input_buf.size());
    Fft<double> fft(13);
    meter.measure([&]() { fft.fft(input_buf, output_buf); });
  };
}

TEST_CASE("fftlib benchmark (float)", "[!benchmark]") {
  BENCHMARK_ADVANCED("fft 8")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(8);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(3);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
  BENCHMARK_ADVANCED("fft 32")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(32);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(5);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
  BENCHMARK_ADVANCED("fft 64")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(64);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(6);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
  BENCHMARK_ADVANCED("fft 512")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(512);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(9);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
  BENCHMARK_ADVANCED("fft 1024")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(1024);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(10);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
  BENCHMARK_ADVANCED("fft 8192")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(8192);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    Fft<float> fft(13);
    meter.measure([&]() { fft.fft(input_buf.data(), output_buf.data()); });
  };
}
}  // namespace fftlib
