cmake_minimum_required(VERSION 3.1)
PROJECT(fftlib LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Test

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(${PROJECT_NAME}_test fftlib_test.cpp)
target_link_libraries(${PROJECT_NAME}_test
    INTERFACE
    ${PROJECT_NAME}
)
target_include_directories(${PROJECT_NAME}_test PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}_test PRIVATE Catch2::Catch2WithMain)

target_compile_features(${PROJECT_NAME}_test PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(${PROJECT_NAME}_test PRIVATE /W4 /WX)
else()
  target_compile_options(${PROJECT_NAME}_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()

