#include <fftlib.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <iostream>
#include <utility>

namespace fftlib {

template <class T>
void print_vec(const std::vector<T>& vec) {
  std::cout << "{";
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << "{" << vec[i].real() << ", " << vec[i].imag() << "}"
              << ((i == vec.size() - 1) ? "}\n" : ", ");
  }
}

template <class T>
std::vector<T> extract_real(const std::vector<std::complex<T>>& vec) {
  auto n = vec.size();
  std::vector<T> real(n);
  for (size_t i = 0; i < n; i++) {
    real[i] = vec[i].real();
  }
  return real;
}

template <class T>
std::vector<T> extract_imag(const std::vector<std::complex<T>>& vec) {
  auto n = vec.size();
  std::vector<T> imag(n);
  for (size_t i = 0; i < n; i++) {
    imag[i] = vec[i].imag();
  }
  return imag;
}

constexpr double MARGIN = 0.0001;

TEST_CASE("Calculate bit length for FFT", "[bit_length]") {
  REQUIRE(bit_length(0) == 0);
  REQUIRE(bit_length(1) == 1);
  REQUIRE(bit_length(2) == 2);
  REQUIRE(bit_length(4) == 3);
  REQUIRE(bit_length(8) == 4);
}

TEST_CASE("Reverse bits of unsigned integers", "[reverse_bits]") {
  REQUIRE(reverse_bits(0b000, 3) == 0b000);
  REQUIRE(reverse_bits(0b001, 3) == 0b100);
  REQUIRE(reverse_bits(0b011, 3) == 0b110);
  REQUIRE(reverse_bits(0b010, 3) == 0b010);
}

TEST_CASE("FFT 0", "[fft]") {
  std::vector<double> input_buf{0, 1, 0, 1, 0, 1, 0, 1};
  auto output_buf = fft(input_buf);
  print_vec(output_buf);
  std::vector<std::complex<double>> expected{{4., 0.}, {0., 0.},  {0., 0.},
                                             {0., 0.}, {-4., 0.}, {0., 0.},
                                             {0., 0.}, {0., 0.}};
  REQUIRE_THAT(extract_real(output_buf),
               Catch::Matchers::Approx(extract_real(expected)).margin(MARGIN));
  REQUIRE_THAT(extract_imag(output_buf),
               Catch::Matchers::Approx(extract_imag(expected)).margin(MARGIN));
}

TEST_CASE("FFT 1", "[fft]") {
  std::vector<double> input_buf{0, 1, 2, 3, 4, 5, 6, 7};
  auto output_buf = fft(input_buf);
  print_vec(output_buf);
  std::vector<std::complex<double>> expected{
      {28, 0}, {-4, 9.65685},  {-4, 4},  {-4, 1.65685},
      {-4, 0}, {-4, -1.65685}, {-4, -4}, {-4, -9.65685}};
  REQUIRE_THAT(extract_real(output_buf),
               Catch::Matchers::Approx(extract_real(expected)).margin(MARGIN));
  REQUIRE_THAT(extract_imag(output_buf),
               Catch::Matchers::Approx(extract_imag(expected)).margin(MARGIN));
}
}  // namespace fftlib
