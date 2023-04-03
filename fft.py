import numpy as np


def is_power_of_two(x: int):
    return (x & (x - 1)) == 0


def bit_length(x: int):
    length = 0
    while x:
        x >>= 1
        length += 1
    return length


def reverse_bits(x: int, bit_length: int):
    rev = 0
    for _ in range(bit_length):
        rev <<= 1
        if x & 1 == 1:
            rev += 1
        x >>= 1
    return rev


def _fft_simple_helper(input_buf, inverse=False):
    n = len(input_buf)
    if n == 1:
        return input_buf

    output_buf = np.zeros(n, dtype=np.complex128)

    # Even rows
    next_even_input = input_buf[0 : n // 2] + input_buf[n // 2 : n]
    output_buf[0:n:2] = _fft_simple_helper(next_even_input, inverse)

    # odd rows
    angle_sign = -1 if inverse else 1
    w_angle = -1 * np.array(np.array(range(n // 2))) * angle_sign * 2 / n * np.pi * 1j
    w = np.exp(w_angle)  # 'W' in textbooks
    next_odd_input = w * (input_buf[0 : n // 2] - input_buf[n // 2 : n])
    output_buf[1:n:2] = _fft_simple_helper(next_odd_input, inverse)

    return output_buf


def fft_simple(input_buf, inverse=False):
    n = len(input_buf)
    if n == 0:
        raise ValueError("Input buffer is empty")

    if not is_power_of_two(n):
        raise ValueError("Input buffer length should be power of two")

    output_buf = _fft_simple_helper(input_buf, inverse)
    divisor = n if inverse else 1  # 'N' in textbooks
    output_buf = output_buf / divisor
    return output_buf


# Sande–Tukey FFT algorithm (decimation in frequency)
def fft_sande_tukey(input_buf, inverse=False):
    n = len(input_buf)
    if n == 0:
        raise ValueError("Input buffer is empty")

    if not is_power_of_two(n):
        raise ValueError("Input buffer length should be power of two")

    if n == 1:
        return input_buf

    output_buf = np.zeros(n, dtype=np.complex128)
    # Deep copy input
    for i, x in enumerate(input_buf):
        output_buf[i] = x
    index_bit_len = bit_length(n - 1)
    num_loop = index_bit_len

    # Calculate FFT using butterfly operation
    for i in range(num_loop):
        num_group = pow(2, i)
        num_element_per_group = n // num_group
        for j in range(num_group):
            # k0: index for the first half of group
            # k1: index for the second half of group
            k0_start = j * num_element_per_group
            k0_end = j * num_element_per_group + num_element_per_group // 2  # exclusive
            for k0 in range(k0_start, k0_end):
                k1 = k0 + num_element_per_group // 2
                idx = k0 - k0_start  # local index
                x0 = output_buf[k0]
                x1 = output_buf[k1]
                angle_sign = -1 if inverse else 1
                w_angle = -1 * 2 * np.pi / num_element_per_group * 1j * idx * angle_sign
                w = np.exp(w_angle)  # 'W' in textbooks
                output_buf[k0] = x0 + x1
                output_buf[k1] = (x0 - x1) * w

    divisor = n if inverse else 1  # 'N' in textbooks
    output_buf = output_buf / divisor

    # Restore order of output using bit inversion
    for i in range(n):
        j = reverse_bits(i, index_bit_len)
        if i < j:
            # Swap
            output_buf[i], output_buf[j] = output_buf[j], output_buf[i]

    return output_buf


# Cooley–Tukey FFT algorithm (decimation in time)
def fft(input_buf, inverse=False):
    n = len(input_buf)
    if n == 0:
        raise ValueError("Input buffer is empty")

    if not is_power_of_two(n):
        raise ValueError("Input buffer length should be power of two")

    if n == 1:
        return input_buf

    output_buf = np.zeros(n, dtype=np.complex128)
    # Deep copy input
    for i, x in enumerate(input_buf):
        output_buf[i] = x

    index_bit_len = bit_length(n - 1)
    num_loop = index_bit_len

    # Reorder of input using bit inversion
    for i in range(n):
        j = reverse_bits(i, index_bit_len)
        if i < j:
            # Swap
            output_buf[i], output_buf[j] = output_buf[j], output_buf[i]

    # Calculate FFT using butterfly operation
    for i in range(num_loop):
        num_element_per_group = pow(2, i + 1)
        num_group = n // num_element_per_group
        for j in range(num_group):
            # k0: index for the first half of group
            # k1: index for the second half of group
            k0_start = j * num_element_per_group
            k0_end = j * num_element_per_group + num_element_per_group // 2  # exclusive
            for k0 in range(k0_start, k0_end):
                k1 = k0 + num_element_per_group // 2
                idx = k0 - k0_start  # local index
                x0 = output_buf[k0]
                x1 = output_buf[k1]
                angle_sign = -1 if inverse else 1
                w_angle = -1 * 2 * np.pi / num_element_per_group * 1j * idx * angle_sign
                w = np.exp(w_angle)  # 'W' in textbooks
                output_buf[k0] = x0 + w * x1
                output_buf[k1] = x0 - w * x1

    divisor = n if inverse else 1  # 'N' in textbooks
    output_buf = output_buf / divisor

    return output_buf


def _check_fft_result(fft_func, input_buf):
    test_case_str = f"{fft_func.__name__}: {input_buf}"
    output_buf = fft_func(input_buf)
    output_np = np.fft.fft(input_buf)
    assert np.allclose(output_buf, output_np), (
        test_case_str + f":\n{output_buf},\n{output_np}"
    )
    assert np.allclose(fft_func(output_buf, True), np.fft.ifft(output_np)), (
        test_case_str + f": {fft_func(output_buf, True)}, {np.fft.ifft(output_np)}"
    )
    assert np.allclose(fft_func(output_buf, True), input_buf), test_case_str


def _test_fft_function(fft_func):
    # Empty input
    try:
        fft_func([])
        raise AssertionError(f"{fft_func.__name__}: Sould throw ValueError")
    except ValueError:
        pass

    # Input with non-power-of-two length
    try:
        fft_func([0, 1, 2])
        raise AssertionError(f"{fft_func.__name__}: Sould throw ValueError")
    except ValueError:
        pass

    _check_fft_result(fft_func, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    _check_fft_result(fft_func, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    _check_fft_result(fft_func, np.array([0, 1, 2, 3]))
    _check_fft_result(fft_func, np.array([1, 1]))
    _check_fft_result(fft_func, np.array([1]))
    _check_fft_result(fft_func, np.array(range(32)))
    # Just some random large array
    _check_fft_result(fft_func, np.array(range(512)) - 256)


# Print complex array in a format which is easy to use in C++
# If you want to use for std::complex<float>, use suffix="f"
def _print_complex_arr_cpp_format(arr, suffix=""):
    print("{", end="")
    for i, x in enumerate(arr):
        print("{", end="")
        print(f"{x.real}{suffix}, {x.imag}{suffix}", end="")
        print("}" + ("" if i == len(arr) - 1 else ", "), end="")
    print("}")


if __name__ == "__main__":
    # Test for helper functions
    assert bit_length(0) == 0
    assert bit_length(1) == 1
    assert bit_length(2) == 2
    assert bit_length(3) == 2
    assert bit_length(4) == 3
    assert bit_length(7) == 3
    assert bit_length(8) == 4
    assert bit_length(31) == 5
    assert bit_length(32) == 6

    assert reverse_bits(0b000, 3) == 0b000
    assert reverse_bits(0b001, 3) == 0b100
    assert reverse_bits(0b011, 3) == 0b110
    assert reverse_bits(0b010, 3) == 0b010
    assert reverse_bits(0b01001, 5) == 0b10010

    _test_fft_function(fft_simple)
    _test_fft_function(fft_sande_tukey)
    _test_fft_function(fft)

    # Use following to add new test cases to C++ code
    # _print_complex_arr_cpp_format(np.fft.fft(np.array(range(32))), "f")
    # _print_complex_arr_cpp_format(np.fft.fft(np.array(range(32))))
    # _print_complex_arr_cpp_format(np.array(range(32)), ".")

    print("OK")
