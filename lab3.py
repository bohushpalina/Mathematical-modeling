# Лабораторная работа №3. Моделирование непрерывных СВ
# Вариант 1: N(0, 9); хи-квадрат, m = 4; Стьюдента, m = 6

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 1000          # объем выборки
ALPHA = 0.05      # уровень значимости
R = 10            # число интервалов для критерия Пирсона
M_SIM = 2000      # число симуляций для ошибки I рода

# Параметры варианта
MU, SIGMA2 = 0, 9
CHI_M = 4
T_M = 6
SIGMA = math.sqrt(SIGMA2)


# Генератор БСВ (линейный конгруэнтный)
class LCG:
    def __init__(self, seed=12345):
        self.state = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2 ** 32

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        # (state + 0.5) / m, чтобы не получить ровно 0 под логарифмом
        return (self.state + 0.5) / self.m


lcg = LCG()


# Стандартная нормальная СВ - метод Бокса-Мюллера
def std_normal_sample(size):
    result = []
    while len(result) < size:
        u1 = lcg.random()
        u2 = lcg.random()
        r = math.sqrt(-2 * math.log(u1))
        result.append(r * math.cos(2 * math.pi * u2))
        result.append(r * math.sin(2 * math.pi * u2))
    return result[:size]


# N(m, s^2): x = m + s * z
def normal_sample(size):
    return [MU + SIGMA * z for z in std_normal_sample(size)]


# Хи-квадрат с k степенями свободы (k четное):
# chi2_k = -2 * ln(u1 * u2 * ... * u_{k/2})
def chi2_sample(size, k=CHI_M):
    result = []
    for _ in range(size):
        prod = 1.0
        for _ in range(k // 2):
            prod *= lcg.random()
        result.append(-2 * math.log(prod))
    return result


# Стьюдент с k степенями свободы: t = z / sqrt(chi2_k / k)
def student_sample(size, k=T_M):
    z = std_normal_sample(size)
    chi = chi2_sample(size, k)
    return [z[i] / math.sqrt(chi[i] / k) for i in range(size)]


# Распределения (теоретические) для критериев
DISTS = {
    "N(0, 9)": stats.norm(loc=MU, scale=SIGMA),
    f"chi2({CHI_M})": stats.chi2(CHI_M),
    f"t({T_M})": stats.t(T_M),
}
GENERATORS = {
    "N(0, 9)": normal_sample,
    f"chi2({CHI_M})": chi2_sample,
    f"t({T_M})": student_sample,
}
# Истинные характеристики (математическое ожидание, дисперсия)
TRUE_CHAR = {
    "N(0, 9)": (MU, SIGMA2),
    f"chi2({CHI_M})": (CHI_M, 2 * CHI_M),
    f"t({T_M})": (0, T_M / (T_M - 2)),
}


def mean_est(x):
    return sum(x) / len(x)


def var_est(x):
    m = mean_est(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


# Критерий Колмогорова: возвращает (статистика sqrt(n)*Dn, отвергается ли H0)
def kolmogorov_test(sample, dist, alpha=ALPHA):
    x = np.sort(np.array(sample))
    n = len(x)
    cdf = dist.cdf(x)
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - cdf), np.max(cdf - (i - 1) / n))
    stat = math.sqrt(n) * d
    critical = stats.kstwobign.ppf(1 - alpha)   # около 1.36
    return stat, stat >= critical


# Критерий Пирсона: R равновероятных интервалов по квантилям H0
def pearson_test(sample, dist, alpha=ALPHA, r=R):
    n = len(sample)
    bounds = dist.ppf(np.arange(1, r) / r)
    idx = np.searchsorted(bounds, np.array(sample))
    observed = np.bincount(idx, minlength=r)
    expected = n / r
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    critical = stats.chi2.ppf(1 - alpha, r - 1)
    return chi2_stat, chi2_stat > critical


# 1-2. Моделирование и оценки
print("1. МОДЕЛИРОВАНИЕ И ОЦЕНКИ ПАРАМЕТРОВ\n")
samples = {}
for name, gen in GENERATORS.items():
    samples[name] = gen(N)
    m_true, v_true = TRUE_CHAR[name]
    m_emp = mean_est(samples[name])
    v_emp = var_est(samples[name])
    print(name)
    print(f"  Мат. ожидание: истинное = {m_true:.4f}, оценка = {m_emp:.4f}, разница = {abs(m_true - m_emp):.4f}")
    print(f"  Дисперсия:     истинная = {v_true:.4f}, оценка = {v_emp:.4f}, разница = {abs(v_true - v_emp):.4f}\n")

# Графики: гистограммы + теоретические плотности
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, smp) in zip(axes, samples.items()):
    ax.hist(smp, bins=30, density=True, edgecolor='black', alpha=0.7, label='Выборка')
    xs = np.linspace(min(smp), max(smp), 400)
    ax.plot(xs, DISTS[name].pdf(xs), 'r-', linewidth=2, label='Теор. плотность')
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.show()

# 5. Проверка каждой выборки каждым критерием
print("2. ПЕРЕКРЕСТНАЯ ПРОВЕРКА (каждая выборка каждым критерием)\n")
print(f"Критическое значение Колмогорова: {stats.kstwobign.ppf(1 - ALPHA):.4f}")
print(f"Критическое значение Пирсона (df = {R - 1}): {stats.chi2.ppf(1 - ALPHA, R - 1):.4f}\n")
for s_name, smp in samples.items():
    for h_name, dist in DISTS.items():
        k_stat, k_rej = kolmogorov_test(smp, dist)
        p_stat, p_rej = pearson_test(smp, dist)
        print(f"Выборка {s_name:8s} | H0 = {h_name:8s} | "
              f"Колмогоров: {k_stat:7.4f} {'ОТВЕРГАЕТСЯ' if k_rej else 'принимается'} | "
              f"Пирсон: {p_stat:9.3f} {'ОТВЕРГАЕТСЯ' if p_rej else 'принимается'}")
    print()

# 3-4. Ошибка I рода
print("3. ИССЛЕДОВАНИЕ ОШИБКИ I РОДА\n")
print(f"Уровень значимости: {ALPHA}, число симуляций: {M_SIM}\n")
for name, gen in GENERATORS.items():
    rej_k = 0
    rej_p = 0
    for _ in range(M_SIM):
        smp = gen(N)
        if kolmogorov_test(smp, DISTS[name])[1]:
            rej_k += 1
        if pearson_test(smp, DISTS[name])[1]:
            rej_p += 1
    print(f"{name:8s}: Колмогоров = {rej_k / M_SIM:.4f}, Пирсон = {rej_p / M_SIM:.4f}")
