from math import cos, sin
from matplotlib import pyplot as plt

TWO_PI = 6.28318530718


def make_omega(n, k, inverse=False):
    theta = -k*TWO_PI/n
    if inverse:
        theta = -theta

    cos_theta = cos(theta)
    sin_theta = sin(theta)

    # complex exponential multiplication:
    def rotate(x, y):
        return (x*cos_theta - y*sin_theta, x*sin_theta + y*cos_theta)

    return rotate


def find_power_of_two(n):
    if n < 2:
        raise Exception(f"Not enough data: {n} data points.")
    a = 1
    while True:
        res = 2**a
        if res >= n:
            return res, res - n
        else:
            a += 1


def fft(v):
    n = len(v)
    n, remainder = find_power_of_two(n)
    v = [(z, 0) for z in v]
    # pad input with zeros up to power of two:
    pad_left = remainder // 2
    pad_right = remainder - pad_left
    print(pad_left, pad_right)
    v = [(0, 0)]*pad_left + v + [(0, 0)]*pad_right
    return v, _fft(v)


def ifft(v):
    n = len(v)
    transformed = _fft(v, True)
    a = 1.0/n
    transformed = [(a*z[0], a*z[1]) for z in transformed]
    # WHY DO WE HAVE TO DO .reverse() TO RECOVER THE ORIGINAL SIGNAL? (IF WE DON'T DO THIS, IT DOESN'T LINE UP)
    # IS MY CODE WRONG???
    transformed.reverse()
    return transformed


def _fft(v, inverse=False):
    n = len(v)
    if n == 1:
        return v
    v_e, v_o = v[::2], v[1::2]
    y_e, y_o = _fft(v_e), _fft(v_o)
    y = [(0, 0)]*n
    for k in range(n//2):
        omega_k = make_omega(n, k, inverse)
        omegak_yok = omega_k(*y_o[k])
        y[k] = y_e[k][0] + omegak_yok[0], y_e[k][1] + omegak_yok[1]
        y[k + n//2] = y_e[k][0] - omegak_yok[0], y_e[k][1] - omegak_yok[1]
    return y


def main():
    signal = [cos(x*TWO_PI/128.0) * sin(x*TWO_PI/512.0) for x in range(1300)]
    padded_signal, transformed = fft(signal)
    recovered = ifft(transformed)
    transformed_real = [z[0] for z in transformed]
    recovered_real = [z[0] for z in recovered]
    plt.ylim(-1.0, 1.0)
    plt.plot(transformed_real)
    plt.plot(padded_signal)
    plt.plot(recovered_real)


if __name__ == '__main__':
    main()
