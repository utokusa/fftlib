#include <fftlib.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <iostream>
#include <utility>

namespace fftlib {

template <class T>
void printVec(const std::vector<T>& vec) {
  std::cout << "{";
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << "{" << vec[i].real() << ", " << vec[i].imag() << "}"
              << ((i == vec.size() - 1) ? "}\n" : ", ");
  }
}

template <class T>
std::vector<T> extractReal(const std::vector<std::complex<T>>& vec) {
  auto n = vec.size();
  std::vector<T> real(n);
  for (size_t i = 0; i < n; i++) {
    real[i] = vec[i].real();
  }
  return real;
}

template <class T>
std::vector<T> extractImag(const std::vector<std::complex<T>>& vec) {
  auto n = vec.size();
  std::vector<T> imag(n);
  for (size_t i = 0; i < n; i++) {
    imag[i] = vec[i].imag();
  }
  return imag;
}

constexpr double MARGIN = 0.0001;

TEST_CASE("Calculate bit length for FFT", "[bitLength]") {
  REQUIRE(bitLength(0) == 0);
  REQUIRE(bitLength(1) == 1);
  REQUIRE(bitLength(2) == 2);
  REQUIRE(bitLength(4) == 3);
  REQUIRE(bitLength(8) == 4);
}

TEST_CASE("Reverse bits of unsigned integers", "[reverseBits]") {
  REQUIRE(reverseBits(0b000, 3) == 0b000);
  REQUIRE(reverseBits(0b001, 3) == 0b100);
  REQUIRE(reverseBits(0b011, 3) == 0b110);
  REQUIRE(reverseBits(0b010, 3) == 0b010);
}

TEST_CASE("FFT 0", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 0, 1, 0, 1, 0, 1};
  Fft<double> fft(3);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{{4., 0.}, {0., 0.},  {0., 0.},
                                             {0., 0.}, {-4., 0.}, {0., 0.},
                                             {0., 0.}, {0., 0.}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 1", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2, 3, 4, 5, 6, 7};
  Fft<double> fft(3);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{
      {28, 0}, {-4, 9.65685},  {-4, 4},  {-4, 1.65685},
      {-4, 0}, {-4, -1.65685}, {-4, -4}, {-4, -9.65685}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 2", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2, 3};
  Fft<double> fft(2);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{
      {6, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 3", "[fft]") {
  std::vector<std::complex<double>> input_buf{1, 1};
  Fft<double> fft(1);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{{2, 0}, {0, 0}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 4", "[fft]") {
  std::vector<std::complex<double>> input_buf{1};
  Fft<double> fft(0);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{1};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 5 (Empty)", "[fft]") {
  std::vector<std::complex<double>> input_buf{};
  Fft<double> fft(3);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 6 (Non-power-of-two length)", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2};
  Fft<double> fft(3);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
}

TEST_CASE("FFT 6 (float)", "[fft]") {
  std::vector<std::complex<float>> input_buf{0, 1, 2, 3, 4, 5, 6, 7};
  Fft<float> fft(3);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<float>> expected{
      {28.f, 0.f}, {-4.f, 9.65685f},  {-4.f, 4.f},  {-4.f, 1.65685f},
      {-4.f, 0.f}, {-4.f, -1.65685f}, {-4.f, -4.f}, {-4.f, -9.65685f}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected))
                   .margin(static_cast<float>(MARGIN)));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected))
                   .margin(static_cast<float>(MARGIN)));
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed))
                   .margin(static_cast<float>(MARGIN)));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed))
                   .margin(static_cast<float>(MARGIN)));
}
}  // namespace fftlib
