import random
import math
import numpy as np
from scipy import stats

np.random.normal()

m = 0
s2 = 1
a = 2
w = math.exp(s2)
K = 10


def norm_dist():
    val = sum([random.random() for _ in range(12)]) - 6
    return val


def lognormvariate():
    return math.exp(norm_dist())


def exp_dist(lambd):
    val = (-1 / lambd) * math.log(random.random())
    return val


def generate_sample(gen_func, n, args):
    sample = list()

    for _ in range(n):
        sample.append(gen_func(*args))

    return sample


def expected_math_value(sample):
    tmp = np.unique(sample, return_counts=True)
    values, probabilities = tmp[0], tmp[1] / len(sample)
    math_value = sum(values * probabilities)
    return math_value


def dispersion(sample):
    math_val = expected_math_value(sample)

    tmp = np.unique(sample, return_counts=True)
    values, probabilities = tmp[0], tmp[1] / len(sample)

    disp = sum(probabilities * ((values - math_val) ** 2))
    return disp


def chi_square_test(sample, k):
    def _float_range(start, end, step):
        r = start
        while r < end:
            yield r
            r += step

    interv = (max(sample) - min(sample)) / k
    counts = []
    for i in _float_range(min(sample), max(sample), interv):
        count = list(map(lambda x: i <= x < i + interv, sample)).count(True)
        counts.append(count)

    chi = sum([((c / len(sample) - 1 / k) ** 2) / (1 / k) for c in counts])
    return chi


def kolmogorov_test(sample):
    n = len(sample)

    Dn_plus = max([(i + 1) / n - stats.norm.cdf(x) for i, x in enumerate(sorted(sample))])  # cdf - lognorm func
    Dn_minus = max([stats.norm.cdf(x) - i / n for i, x in enumerate(sorted(sample))])
    Dn = max(Dn_plus, Dn_minus)

    return Dn


if __name__ == '__main__':
    norm = generate_sample(norm_dist, 1000, ())
    lognorm = generate_sample(lognormvariate, 1000, ())
    exp = generate_sample(exp_dist, 1000, (a,))

    print('Стандартное нормальное')
    print(f'Maт. ожидание = {expected_math_value(norm)}. Истинное = {m}')
    print(f'Дисперсия = {dispersion(norm)}. Истинная = {s2}')
    print(f'Критерий Пирсона: {chi_square_test(norm, K)}')
    print(f'Критерий Колмогорова: {kolmogorov_test(norm)}')
    print(np.histogram(norm))
    print()

    print('Логнормальное')
    print(f'Maт. ожидание = {expected_math_value(lognorm)}. Истинное = {1}')
    print(f'Дисперсия = {dispersion(lognorm)}. Истинная = {0}')
    print(f'Критерий Пирсона: {chi_square_test(lognorm, K)}')
    print(f'Критерий Колмогорова: {kolmogorov_test(lognorm)}')
    print()

    print('Экспоненциальное')
    print(f'Maт. ожидание = {expected_math_value(exp)}. Истинное = {1 / a}')
    print(f'Дисперсия = {dispersion(exp)}. Истинная = {1 / (a ** 2)}')
    print(f'Критерий Пирсона: {chi_square_test(exp, K)}')
    print(f'Критерий Колмогорова: {kolmogorov_test(exp)}')
    print()
