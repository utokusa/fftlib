CONFIG?=Debug
BUILD_DIR?=build
GENERATOR_OPTION?=-G "Ninja"

# Preparation (Unix):
# First run `.\python\prepare_env.sh` then `source python/env/bin/activate`.

# Preparation (Windows):
# We use WSL for python related commands (lint, lint-fix, test-python).
# WSL need clang-format and make command: `sudo apt update && sudo apt install clang-format make -y`
# It also requires the python venv: `source python/env/bin/activate`

ifeq ($(OS), Windows_NT)
	# Windows
	TESTS_BIN_PATH:=.\${BUILD_DIR}\${CONFIG}\fftlib_test.exe
	BENCHMARK_BIN_PATH:=.\${BUILD_DIR}\${CONFIG}\fftlib_benchmark.exe
	BENCHMARK_JUCE_BIN_PATH:=.\${BUILD_DIR}\fftlib_juce_benchmark_artefacts\${CONFIG}\fftlib_juce_benchmark.exe
	BENCHMARK_FLAG:=\[\!benchmark\]
	RM_COMMAND=rmdir /s /q ${BUILD_DIR}
	# Use default just because I don't know how to use Ninja in Windows 
	GENERATOR_OPTION=
else
	TESTS_BIN_PATH:=./${BUILD_DIR}/fftlib_test
	BENCHMARK_BIN_PATH:=./${BUILD_DIR}/fftlib_benchmark
	BENCHMARK_JUCE_BIN_PATH:=./${BUILD_DIR}/fftlib_juce_benchmark_artefacts/${CONFIG}/fftlib_juce_benchmark
	BENCHMARK_FLAG:="[!benchmark]"
	RM_COMMAND=rm -rf ${BUILD_DIR}
endif

configure:
	cmake . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) ${GENERATOR_OPTION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

configure-extra:
	cmake . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) ${GENERATOR_OPTION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXTRA_BENCH=ON

build-cpp:
	cmake --build $(BUILD_DIR) --config $(CONFIG)

.PHONY: lint
lint:
	git ls-files | grep -e '\.py$$' | xargs python -m pylint && ./lint.sh

.PHONY: lint-fix
lint-fix:
	git ls-files | grep -e '\.py$$' | xargs python -m black && ./lint.sh fix

.PHONY: test-cpp
test-cpp: build-cpp
	${TESTS_BIN_PATH}

.PHONY: test-python
test-python:
	python3 fft.py

.PHONY: test
test:
	make test-python && make test-cpp

.PHONY: bench
bench: build-cpp
	${BENCHMARK_BIN_PATH} ${BENCHMARK_FLAG}

.PHONY: bench-extra
bench-extra: build-cpp
	${BENCHMARK_JUCE_BIN_PATH} ${BENCHMARK_FLAG}

.PHONY: check-all
check-all:
	make lint && make test && make bench

.PHONY: check-all-extra
check-all-extra:
	make lint && make test && make bench && make bench-extra

# For Windows. Run these from powershell.

.PHONY: win-lint
win-lint:
	wsl bash -c "source python/env/bin/activate; make lint"

.PHONY: win-lint-fix
win-lint-fix:
	wsl bash -c "source python/env/bin/activate; make lint-fix"

.PHONY: win-test
win-test:
	wsl bash -c "source python/env/bin/activate; make test-python"
	make test-cpp

.PHONY: win-check-all
win-check-all:
	make win-lint
	make win-test
	make bench

.PHONY: win-check-all-extra
win-check-all-extra:
	make win-check-all && make bench-extra

.PHONY: clean
clean:
	${RM_COMMAND}

