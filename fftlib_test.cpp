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

TEST_CASE("FFT 7 (float)", "[fft]") {
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

TEST_CASE("FFT 8 (Overload1)", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::complex<double>> output_buf(input_buf.size());
  Fft<double> fft(3);
  REQUIRE(fft.fft(input_buf, output_buf));
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{
      {28, 0}, {-4, 9.65685},  {-4, 4},  {-4, 1.65685},
      {-4, 0}, {-4, -1.65685}, {-4, -4}, {-4, -9.65685}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  std::vector<std::complex<double>> reversed(input_buf.size());
  REQUIRE(fft.fft(output_buf, reversed, true));
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 9 (Overload2)", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::complex<double>> output_buf(input_buf.size());
  Fft<double> fft(3);
  REQUIRE(fft.fft(input_buf.data(), output_buf.data()));
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{
      {28, 0}, {-4, 9.65685},  {-4, 4},  {-4, 1.65685},
      {-4, 0}, {-4, -1.65685}, {-4, -4}, {-4, -9.65685}};
  REQUIRE_THAT(extractReal(output_buf),
               Catch::Matchers::Approx(extractReal(expected)).margin(MARGIN));
  REQUIRE_THAT(extractImag(output_buf),
               Catch::Matchers::Approx(extractImag(expected)).margin(MARGIN));
  std::vector<std::complex<double>> reversed(input_buf.size());
  REQUIRE(fft.fft(output_buf.data(), reversed.data(), true));
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 10 (Empty, Overload1)", "[fft]") {
  std::vector<std::complex<double>> input_buf{};
  std::vector<std::complex<double>> output_buf(input_buf.size());
  Fft<double> fft(3);
  REQUIRE_FALSE(fft.fft(input_buf, output_buf));
  std::vector<std::complex<double>> reversed(input_buf.size());
  REQUIRE_FALSE(fft.fft(output_buf, reversed, true));
}

TEST_CASE("FFT 11 (Non-power-of-two length, Overload1)", "[fft]") {
  std::vector<std::complex<double>> input_buf{0, 1, 2};
  std::vector<std::complex<double>> output_buf(input_buf.size());
  Fft<double> fft(3);
  REQUIRE_FALSE(fft.fft(input_buf, output_buf));
}

TEST_CASE("FFT 12 (length=32, double)", "[fft]") {
  std::vector<std::complex<double>> input_buf{
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  Fft<double> fft(5);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<double>> expected{{496.0, 0.0},
                                             {-16.0, 162.45072620174176},
                                             {-16.0, 80.43743187401357},
                                             {-16.0, 52.744931343013135},
                                             {-16.0, 38.62741699796952},
                                             {-16.0, 29.933894588630228},
                                             {-16.0, 23.945692202647823},
                                             {-16.0, 19.49605640940762},
                                             {-16.0, 16.0},
                                             {-16.0, 13.130860653258562},
                                             {-16.0, 10.690858206708779},
                                             {-16.0, 8.55217817521266},
                                             {-16.0, 6.627416997969522},
                                             {-16.0, 4.853546937717486},
                                             {-16.0, 3.1825978780745316},
                                             {-16.0, 1.5758624537146204},
                                             {-16.0, 0.0},
                                             {-16.0, -1.5758624537146346},
                                             {-16.0, -3.1825978780745316},
                                             {-16.0, -4.853546937717471},
                                             {-16.0, -6.627416997969522},
                                             {-16.0, -8.55217817521267},
                                             {-16.0, -10.690858206708779},
                                             {-16.0, -13.130860653258555},
                                             {-16.0, -16.0},
                                             {-16.0, -19.496056409407625},
                                             {-16.0, -23.945692202647823},
                                             {-16.0, -29.93389458863023},
                                             {-16.0, -38.62741699796952},
                                             {-16.0, -52.744931343013135},
                                             {-16.0, -80.43743187401357},
                                             {-16.0, -162.45072620174176}};
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

TEST_CASE("FFT 13 (length=32, float)", "[fft]") {
  std::vector<std::complex<float>> input_buf{
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  Fft<float> fft(5);
  auto output_buf = fft.fft(input_buf);
  // printVec(output_buf);
  std::vector<std::complex<float>> expected{{496.0f, 0.0f},
                                            {-16.0f, 162.45072620174176f},
                                            {-16.0f, 80.43743187401357f},
                                            {-16.0f, 52.744931343013135f},
                                            {-16.0f, 38.62741699796952f},
                                            {-16.0f, 29.933894588630228f},
                                            {-16.0f, 23.945692202647823f},
                                            {-16.0f, 19.49605640940762f},
                                            {-16.0f, 16.0f},
                                            {-16.0f, 13.130860653258562f},
                                            {-16.0f, 10.690858206708779f},
                                            {-16.0f, 8.55217817521266f},
                                            {-16.0f, 6.627416997969522f},
                                            {-16.0f, 4.853546937717486f},
                                            {-16.0f, 3.1825978780745316f},
                                            {-16.0f, 1.5758624537146204f},
                                            {-16.0f, 0.0f},
                                            {-16.0f, -1.5758624537146346f},
                                            {-16.0f, -3.1825978780745316f},
                                            {-16.0f, -4.853546937717471f},
                                            {-16.0f, -6.627416997969522f},
                                            {-16.0f, -8.55217817521267f},
                                            {-16.0f, -10.690858206708779f},
                                            {-16.0f, -13.130860653258555f},
                                            {-16.0f, -16.0f},
                                            {-16.0f, -19.496056409407625f},
                                            {-16.0f, -23.945692202647823f},
                                            {-16.0f, -29.93389458863023f},
                                            {-16.0f, -38.62741699796952f},
                                            {-16.0f, -52.744931343013135f},
                                            {-16.0f, -80.43743187401357f},
                                            {-16.0f, -162.45072620174176f}};
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

TEST_CASE("FFT 14 (large array, double)", "[fft]") {
  std::vector<std::complex<double>> input_buf(512);  // 2 ** 9 = 512
  // Create large array with some values
  for (size_t i = 0; i < input_buf.size(); i++) {
    input_buf[i] = static_cast<double>(i) - 256.0;
  }
  Fft<double> fft(9);
  auto output_buf = fft.fft(input_buf);
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}

TEST_CASE("FFT 15 (large array, float)", "[fft]") {
  std::vector<std::complex<float>> input_buf(512);  // 2 ** 9 = 512
  // Create large array with some values
  for (size_t i = 0; i < input_buf.size(); i++) {
    input_buf[i] = static_cast<float>(i) - 256.0;
  }
  Fft<float> fft(9);
  auto output_buf = fft.fft(input_buf);
  auto reversed = fft.fft(output_buf, true);
  REQUIRE_THAT(extractReal(input_buf),
               Catch::Matchers::Approx(extractReal(reversed)).margin(MARGIN));
  REQUIRE_THAT(extractImag(input_buf),
               Catch::Matchers::Approx(extractImag(reversed)).margin(MARGIN));
}
}  // namespace fftlib
