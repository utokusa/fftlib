import numpy as np


def is_power_of_two(x: int):
    return (x & (x - 1)) == 0


def fft(input_buf, inverse=False):
    n = len(input_buf)
    if n == 0:
        raise ValueError("Input buffer is empty")

    if not is_power_of_two(n):
        raise ValueError("Input buffer length should be power of two")

    if n == 1:
        return input_buf

    output_buf = np.zeros(n, dtype=np.complex128)

    # Even rows
    next_even_input = input_buf[0 : n // 2] + input_buf[n // 2 : n]
    output_buf[0:n:2] = fft(next_even_input)

    # odd rows
    angle_sign = -1 if inverse else 1
    w_angle = -1 * np.array(np.array(range(n // 2))) * angle_sign * 2 / n * np.pi * 1j
    w = np.exp(w_angle)  # 'W' in textbooks
    divisor = n if inverse else 1  # 'N' in textbooks
    next_odd_input = w / divisor * (input_buf[0 : n // 2] - input_buf[n // 2 : n])
    output_buf[1:n:2] = fft(next_odd_input)

    return output_buf


if __name__ == "__main__":
    # Test
    # Normal case
    input_buf = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    output_buf = fft(input_buf)
    output_np = np.fft.fft(input_buf)
    assert np.allclose(output_buf, output_np)
    assert np.allclose(fft(output_buf, True), np.fft.ifft(output_np))
    assert np.allclose(fft(output_buf, True), input_buf)

    # Empty input
    try:
        fft([])
        raise AssertionError("Sould throw ValueError")
    except ValueError:
        pass

    # Input with non-power-of-two length
    try:
        fft([0, 1, 2])
        raise AssertionError("Sould throw ValueError")
    except ValueError:
        pass

    print("OK")
