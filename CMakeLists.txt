cmake_minimum_required(VERSION 3.1)
PROJECT(fftlib LANGUAGES CXX)

set(FFTLIB_CPP_STANDARD cxx_std_20)

function(add_common_compile_option target_name)
  target_compile_features(${target_name} PRIVATE ${FFTLIB_CPP_STANDARD})
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
endfunction()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(${PROJECT_NAME} INTERFACE ${FFTLIB_CPP_STANDARD})

# Test dependency

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# fftlib test
set(FFTLIB_TEST_NAME "${PROJECT_NAME}_test")
add_executable(${FFTLIB_TEST_NAME} fftlib_test.cpp)
target_link_libraries(${FFTLIB_TEST_NAME}
  INTERFACE
  ${PROJECT_NAME}
)
target_include_directories(${FFTLIB_TEST_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${FFTLIB_TEST_NAME} PRIVATE Catch2::Catch2WithMain)
add_common_compile_option(${FFTLIB_TEST_NAME})


# fftlib benchmark
set(FFTLIB_BENCHMARK_NAME "${PROJECT_NAME}_benchmark")
add_executable(${FFTLIB_BENCHMARK_NAME} fftlib_benchmark.cpp)
target_link_libraries(${FFTLIB_BENCHMARK_NAME}
  INTERFACE
  ${PROJECT_NAME}
)
target_include_directories(${FFTLIB_BENCHMARK_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${FFTLIB_BENCHMARK_NAME} PRIVATE Catch2::Catch2WithMain)
add_common_compile_option(${FFTLIB_BENCHMARK_NAME})
