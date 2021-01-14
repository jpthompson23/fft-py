import math
from math import cos, sin
from matplotlib import pyplot as plt

TWO_PI = 6.28318530718


def make_omega(n, k, is_forward):
    theta = -k*TWO_PI/n
    if not is_forward:
        # inverse transform:
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


def make_gauss(mu, sigma):
    sigma_squared = sigma**2
    return lambda x: math.exp(-0.5*(x - mu)**2 / sigma_squared)


def gaussian_padding(v, fade_in=0.10):
    n = len(v)
    half_n = n/2.0
    # after fade-in we should be at some number of sigma:
    sigma = (1.0 - fade_in) * half_n / 1.5
    gauss = make_gauss(half_n, sigma)
    # print([gauss(i) for i in range(n)])
    gauss_v = [(gauss(i)*value, 0) for i, value in enumerate(v)]
    n, remainder = find_power_of_two(n)
    pad_left = remainder // 2
    pad_right = remainder - pad_left
    new_v = [(0, 0)]*pad_left + gauss_v + [(0, 0)]*pad_right
    return new_v


def fft(v):
    v = gaussian_padding(v)
    n = len(v)
    t = _fft(v, is_forward=True)
    return v, t


def ifft(v):
    n = len(v)
    # is_forward is False for inverse transform
    transformed = _fft(v, is_forward=False)
    a = 1.0/n
    transformed = [(a*z[0], a*z[1]) for z in transformed]
    return transformed


def _fft(v, is_forward):
    n = len(v)
    if n == 1:
        return v
    v_e, v_o = v[::2], v[1::2]
    y_e, y_o = _fft(v_e, is_forward), _fft(v_o, is_forward)
    y = [(0, 0)]*n
    for k in range(n//2):
        omega_k = make_omega(n, k, is_forward)
        omegak_yok = omega_k(*y_o[k])
        y[k] = y_e[k][0] + omegak_yok[0], y_e[k][1] + omegak_yok[1]
        y[k + n//2] = y_e[k][0] - omegak_yok[0], y_e[k][1] - omegak_yok[1]
    return y


def main():
    signal = [cos(x*TWO_PI/32.0) for x in range(900)]
    # signal = [1.0 for x in range(1900)]
    padded_signal, t = fft(signal)
    recovered = ifft(t)
    n = len(t)
    t_left = t[n//2:0:-1]
    t_right = t[n:n//2:-1]
    transformed = t_left + t_right
    transformed_mag = [math.sqrt(z[0]**2 + z[1]**2) for z in transformed]
    recovered_real = [z[0] for z in recovered]
    # print(transformed_mag)
    plt.ylim(-1.0, 1.0)
    plt.plot(padded_signal)
    plt.plot(transformed_mag)
    plt.plot(recovered_real)


if __name__ == '__main__':
    main()