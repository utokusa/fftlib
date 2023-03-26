CONFIG?=Debug
BUILD_DIR?=build
GENERATOR_OPTION?=-G "Ninja"

configure:
	cmake . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) ${GENERATOR_OPTION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

configure-extra:
	cmake . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) ${GENERATOR_OPTION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXTRA_BENCH=ON

build-cpp:
	cmake --build $(BUILD_DIR) --config $(CONFIG)

.PHONY: lint
lint:
	git ls-files | grep -e '\.py$$' | xargs pylint && ./lint.sh

.PHONY: lint-fix
lint-fix:
	git ls-files | grep -e '\.py$$' | xargs black && ./lint.sh fix

.PHONY: test-cpp
test-cpp: build-cpp
	./$(BUILD_DIR)/fftlib_test

.PHONY: test-python
test-python:
	python3 fft.py

.PHONY: test
test:
	make test-python && make test-cpp

.PHONY: bench
bench: build-cpp
	./${BUILD_DIR}/fftlib_benchmark "[!benchmark]"

.PHONY: bench-extra
bench-extra: build-cpp
	./${BUILD_DIR}/fftlib_juce_benchmark_artefacts/${CONFIG}/fftlib_juce_benchmark "[!benchmark]"

.PHONY: check-all
check-all:
	make lint && make test && make bench

.PHONY: check-all-extra
check-all-extra:
	make lint && make test && make bench && make bench-extra
