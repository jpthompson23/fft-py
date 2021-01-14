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
    return lambda x, y: (x*cos_theta - y*sin_theta, x*sin_theta + y*cos_theta)


def find_power_of_two(n):
    log_2_n = math.log2(n)
    pow = math.ceil(log_2_n)
    result = 2**pow
    return result, result - n


def make_gauss(mu, ss, a):
    return lambda x: a*math.exp(-0.5*(x - mu)**2 / ss)


def gaussian_padding(v):
    n = len(v)

    if n < 4:
        raise Exception(f"Not enough data: {n} data points.")

    n, remainder = find_power_of_two(n)
    pad_right = remainder // 2
    pad_left = remainder - pad_right

    # sigma will be some fraction of the padding
    sigma_left = pad_left*0.333
    sigma_right = pad_right*0.333
    ssl = sigma_left**2
    ssr = sigma_right**2

    d_first = v[1] - v[0]
    d_last = v[-2] - v[-1]
    mu_left = ssl * d_first / v[0] + pad_left
    mu_right = ssr * d_last / v[-1] + pad_right
    fit_gauss_l = make_gauss(mu_left, ssl, 1.0)
    fit_gauss_r = make_gauss(mu_right, ssr, 1.0)
    a_left = v[0] / fit_gauss_l(pad_left)
    a_right = v[-1] / fit_gauss_r(pad_right)
    left_gauss = make_gauss(mu_left, ssl, a_left)
    right_gauss = make_gauss(mu_right, ssr, a_right)
    fade_in = [(left_gauss(i), 0) for i in range(pad_left)]
    fade_out = [(right_gauss(i), 0) for i in range(pad_right)]
    fade_out.reverse()

    new_v = fade_in + [(x, 0) for x in v] + fade_out
    return new_v


def fft(v):
    v = gaussian_padding(v)
    n = len(v)
    t = _fft(v, is_forward=True)
    a = 1.0/math.sqrt(n)
    t = [(a*z[0], a*z[1]) for z in t]
    return v, t


def ifft(v):
    n = len(v)
    # is_forward is False for inverse transform
    t = _fft(v, is_forward=False)
    a = 1.0/math.sqrt(n)
    t = [(a*z[0], a*z[1]) for z in t]
    return t


def _fft(v, is_forward):
    n = len(v)
    if n == 1:
        return v
    v_even, v_odd = v[::2], v[1::2]
    t_even, t_odd = _fft(v_even, is_forward), _fft(v_odd, is_forward)
    t_res = [(0, 0)]*n
    for k in range(n//2):
        omega = make_omega(n, k, is_forward)
        omega_t_odd = omega(*t_odd[k])
        t_res[k] = (t_even[k][0] + omega_t_odd[0], t_even[k][1] + omega_t_odd[1])
        t_res[k + n//2] = (t_even[k][0] - omega_t_odd[0], t_even[k][1] - omega_t_odd[1])
    return t_res


def main():
    signal = [cos(x*TWO_PI/160.0) for x in range(350)]
    # signal = [1.0 for x in range(1900)]
    padded_signal, t = fft(signal)
    print(len(padded_signal))
    recovered = ifft(t)
    n = len(t)
    t_left = t[n//2:0:-1]
    t_right = t[n:n//2:-1]
    transformed = t_left + t_right
    transformed_mag = [math.sqrt(z[0]**2 + z[1]**2) for z in transformed]
    recovered_real = [z[0] for z in recovered]
    print(transformed_mag)
    plt.ylim(-1.5, 1.5)
    plt.plot(padded_signal)
    plt.plot(recovered_real)
    plt.plot(transformed_mag)


if __name__ == '__main__':
    main()
