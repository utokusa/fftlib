cmake_minimum_required(VERSION 3.1)
PROJECT(fftlib
  LANGUAGES CXX
  VERSION 0.0.1
  )


option(EXTRA_BENCH "Compile benchmark of 3rd-party FFT libraries" OFF)

set(FFTLIB_CPP_STANDARD cxx_std_20)

function(add_common_compile_option target_name)
  target_compile_features(${target_name} PRIVATE ${FFTLIB_CPP_STANDARD})
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
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

if (EXTRA_BENCH)
  # FFT libraries
  # Add JUCE
  # Use `git clone --recursive <this repo>` or `git submodule update --init --recursive`
  add_subdirectory(lib/JUCE)
  set(JUCE_FFT_BENCHMARK_NAME "${PROJECT_NAME}_juce_benchmark")
  juce_add_console_app(${JUCE_FFT_BENCHMARK_NAME})
  target_compile_definitions(${JUCE_FFT_BENCHMARK_NAME}
    PRIVATE
    # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
    JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_plugin` call
    JUCE_USE_CURL=0)    # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_plugin` call
  target_sources(${JUCE_FFT_BENCHMARK_NAME} PRIVATE
    third-party-bench/juce_fft_benchmark.cpp) # If your target code includes .cpp files, add them here
  target_link_libraries(${JUCE_FFT_BENCHMARK_NAME} PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
        )

  juce_generate_juce_header(${JUCE_FFT_BENCHMARK_NAME})
  target_link_libraries(${JUCE_FFT_BENCHMARK_NAME} PRIVATE Catch2::Catch2WithMain)
  add_common_compile_option(${JUCE_FFT_BENCHMARK_NAME})
endif()
