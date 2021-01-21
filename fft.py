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

    if n < 2:
        raise Exception(f"Not enough data: {n} data points.")

    n, remainder = find_power_of_two(n)
    if remainder == 0:
        fade_in = []
        fade_out = []
    else:
        pad_right = remainder // 2
        pad_left = remainder - pad_right

        # set sigma to be some fraction of the padding:
        sigma = pad_left*0.333
        ss = sigma**2

        tol = 0.001
        d_first = v[1] - v[0]
        d_last = v[-2] - v[-1]
        if v[0] < tol or v[-2] < tol:
            # avoid division by zero when values are small; simply pad with zeros:
            fade_in = [(0, 0) for i in range(pad_left)]
            fade_out = [(0, 0) for i in range(pad_right)]
        else:
            mu_left = math.fabs(ss * d_first / v[0] + pad_left)
            mu_right = math.fabs(ss * d_last / v[-2] + pad_right)
            fit_gauss_l = make_gauss(mu_left, ss, 1.0)
            fit_gauss_r = make_gauss(mu_right, ss, 1.0)
            fit_left = fit_gauss_l(pad_left)
            fit_right = fit_gauss_r(pad_right)

            # avoid dividing by small values when attempting to fit
            if fit_left < tol:
                fit_left = tol
            if fit_right < tol:
                fit_right = tol
            a_left = v[0] / fit_left
            a_right = v[-1] / fit_right
            left_gauss = make_gauss(mu_left, ss, a_left)
            right_gauss = make_gauss(mu_right, ss, a_right)
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
    gauss = make_gauss(450, 64, 0.5)
    signal = [gauss(x) for x in range(900)]
    # signal = [cos(x*TWO_PI/29) for x in range(900)]
    # signal = [1.0 for x in range(1900)]
    padded_signal, t = fft(signal)
    print(len(padded_signal))
    recovered = ifft(t)
    n = len(t)
    t_pos = t[:n//2]
    t_neg = t[n//2:]
    # transpose the negative and positive sides of the fourier transform:
    transformed = t_neg + t_pos
    transformed_mag = [math.sqrt(z[0]**2 + z[1]**2) for z in transformed]
    recovered_real = [z[0] for z in recovered]
    # print(transformed_mag)
    plt.ylim(-1.5, 1.5)
    plt.plot(padded_signal)
    plt.plot(recovered_real)
    plt.plot(transformed_mag)


if __name__ == '__main__':
    main()
