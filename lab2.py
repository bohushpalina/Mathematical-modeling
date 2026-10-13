# Лабораторная работа №2. Моделирование дискретных СВ
# Вариант 1: Бернулли Bi(1, 0.7); биномиальное Bi(5, 0.25)

import math
from bisect import bisect_right
import matplotlib.pyplot as plt
from scipy import stats

N = 1000          # объем выборки
ALPHA = 0.05      # уровень значимости
M_SIM = 10000     # число симуляций для ошибки I рода

# Теоретические законы распределения {значение: вероятность}
P_BER = 0.7
M_BIN, P_BIN = 5, 0.25
BER = {0: 1 - P_BER, 1: P_BER}
BIN = {k: math.comb(M_BIN, k) * P_BIN ** k * (1 - P_BIN) ** (M_BIN - k) for k in range(M_BIN + 1)}
LAWS = {"Бернулли Bi(1, 0.7)": BER, "Биномиальное Bi(5, 0.25)": BIN}


# Линейный конгруэнтный генератор БСВ
class LCG:
    def __init__(self, seed=12345):
        self.state = seed

    def random(self):
        self.state = (1664525 * self.state + 1013904223) % 2 ** 32
        return self.state / 2 ** 32


lcg = LCG()


# Метод обратной функции: x_i, если F(x_{i-1}) <= u < F(x_i)
def generate_discrete(probs, size):
    values = list(probs)
    cdf, acc = [], 0.0
    for p in probs.values():
        acc += p
        cdf.append(acc)
    cdf[-1] = 1.0
    return [values[bisect_right(cdf, lcg.random())] for _ in range(size)]


# Оценки: среднее и несмещенная дисперсия
def mean_var(x):
    m = sum(x) / len(x)
    return m, sum((v - m) ** 2 for v in x) / (len(x) - 1)


# Критерий Пирсона с объединением интервалов (ожидаемая частота >= 5)
def pearson_test(sample, probs, alpha=ALPHA, verbose=False):
    n = len(sample)
    if not set(sample) <= set(probs):          # невозможные значения
        if verbose:
            print("  В выборке есть значения, невозможные при H0 -> ОТВЕРГАЕТСЯ")
        return float('inf'), 0.0, True

    obs, exp, o_acc, e_acc = [], [], 0, 0
    for k in sorted(probs):
        o_acc += sample.count(k)
        e_acc += probs[k] * n
        if e_acc >= 5:
            obs.append(o_acc)
            exp.append(e_acc)
            o_acc = e_acc = 0
    if e_acc > 0:                              # остаток присоединяем к последней группе
        obs[-1] += o_acc
        exp[-1] += e_acc

    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    df = len(exp) - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)
    reject = p_value < alpha
    if verbose:
        print(f"  X^2 = {chi2:.4f}, df = {df}, p-value = {p_value:.4f} -> "
              f"{'ОТВЕРГАЕТСЯ' if reject else 'ПРИНИМАЕТСЯ'}")
    return chi2, p_value, reject


# 1. Моделирование и оценки
print("1. МОДЕЛИРОВАНИЕ И ОЦЕНКА ПАРАМЕТРОВ\n")
samples = {name: generate_discrete(p, N) for name, p in LAWS.items()}
for name, p in LAWS.items():
    m_true = sum(k * v for k, v in p.items())
    v_true = sum(k * k * v for k, v in p.items()) - m_true ** 2
    m_emp, v_emp = mean_var(samples[name])
    print(name)
    print(f"  Мат. ожидание: истинное = {m_true:.4f}, оценка = {m_emp:.4f}, разница = {abs(m_true - m_emp):.4f}")
    print(f"  Дисперсия:     истинная = {v_true:.4f}, оценка = {v_emp:.4f}, разница = {abs(v_true - v_emp):.4f}\n")

# Графики: теоретические и эмпирические вероятности
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, p) in zip(axes, LAWS.items()):
    x = list(p)
    ax.bar([k - 0.15 for k in x], list(p.values()), width=0.3, label='Теоретические', alpha=0.7)
    ax.bar([k + 0.15 for k in x], [samples[name].count(k) / N for k in x], width=0.3, label='Эмпирические', alpha=0.7)
    ax.set_xticks(x)
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.show()

# 2. Перекрестная проверка: каждая выборка каждым критерием
print("2. ПРОВЕРКА КРИТЕРИЕМ ХИ-КВАДРАТ\n")
for s_name, smp in samples.items():
    for h_name, p in LAWS.items():
        print(f"Выборка: {s_name} | H0: {h_name}")
        pearson_test(smp, p, verbose=True)

# 3. Ошибка I рода
print(f"\n3. ИССЛЕДОВАНИЕ ОШИБКИ I РОДА (ε = {ALPHA}, симуляций: {M_SIM})\n")
rejected = {name: 0 for name in LAWS}
for _ in range(M_SIM):
    for name, p in LAWS.items():
        if pearson_test(generate_discrete(p, N), p)[2]:
            rejected[name] += 1
for name in LAWS:
    print(f"Эмпирическая ошибка I рода, {name}: {rejected[name] / M_SIM:.4f}")
