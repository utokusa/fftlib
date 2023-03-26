#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <JuceHeader.h>
#include <iostream>
#include <utility>
#include "../benchmark_util.h"

namespace fftlib {

std::vector<juce::dsp::Complex<float>> convertToJuceComplexVec(std::vector<std::complex<double>> input_buf) {
  auto n = input_buf.size();
  std::vector<juce::dsp::Complex<float>> ret(n);
  for (size_t i = 0; i < n; i++) {
    ret[i] = {static_cast<float>(input_buf[i].real()), static_cast<float>(input_buf[i].imag())};
  }
  return ret;
}

TEST_CASE("JUCE benchmark (float)", "[!benchmark]") {
  BENCHMARK_ADVANCED("fft 8")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(8);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(3);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
  BENCHMARK_ADVANCED("fft 32")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(32);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(5);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
  BENCHMARK_ADVANCED("fft 64")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(64);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(6);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
  BENCHMARK_ADVANCED("fft 512")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(512);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(9);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
  BENCHMARK_ADVANCED("fft 1024")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(1024);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(10);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
  BENCHMARK_ADVANCED("fft 8192")(Catch::Benchmark::Chronometer meter) {
    std::vector<std::complex<float>> input_buf = genRandVec<float>(8192);
    std::vector<std::complex<float>> output_buf(input_buf.size());
    juce::dsp::FFT fft(13);
    meter.measure([&]() { fft.perform(input_buf.data(), output_buf.data(), false); });
  };
}
}  // namespace fftlib

