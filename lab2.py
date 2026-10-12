import math
import matplotlib.pyplot as plt
from scipy import stats

N = 1000
ALPHA = 0.05

# Параметры Бернулли
p_ber = 0.7

# Параметры Биномиального
m_bin = 5
p_bin = 0.25

# 0. Генератор случайных чисел (ЛКГ)
class LCG:
    def __init__(self, seed=42):
        self.state = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32

    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

lcg = LCG(seed=12345)

def uniform_random_sample(size):
    """Генерация массива равномерных случайных чисел от 0 до 1"""
    return [lcg.random() for _ in range(size)]


# 1. Теоретические функции (с использованием math.comb)
def get_bernoulli_probs():
    """Возвращает словарь теоретических вероятностей для Бернулли"""
    return {0: 1 - p_ber, 1: p_ber}

def get_binomial_probs():
    """Возвращает словарь теоретических вероятностей для Биномиального"""
    probs = {}
    for k in range(m_bin + 1):
        # Используем встроенный math.comb для сочетаний
        comb = math.comb(m_bin, k)
        probs[k] = comb * (p_bin ** k) * ((1 - p_bin) ** (m_bin - k))
    return probs

def generate_discrete(probs_dict, size):
    """
    Моделирование дискретной СВ методом обратной функции (инверсии).
    """
    values = list(probs_dict.keys())
    probabilities = list(probs_dict.values())
    
    # Построение кумулятивных вероятностей
    cdf = []
    accumulated = 0.0
    for p in probabilities:
        accumulated += p
        cdf.append(accumulated)
    cdf[-1] = 1.0
    
    uniforms = uniform_random_sample(size)
    sample = []
    
    for u in uniforms:
        chosen_val = values[-1]
        for idx, bound in enumerate(cdf):
            if u < bound:
                chosen_val = values[idx]
                break
        sample.append(chosen_val)
        
    return sample


# 2. Ручной расчет статистик выборки
def calculate_mean(sample):
    return sum(sample) / len(sample)

def calculate_variance(sample, mean_val):
    """Несмещенная оценка дисперсии (ddof=1)"""
    n = len(sample)
    if n <= 1:
        return 0.0
    sq_diff_sum = sum((x - mean_val) ** 2 for x in sample)
    return sq_diff_sum / (n - 1)


# 3. Критерий Пирсона (Хи-квадрат)
def pearson_chi2_test(sample, theor_probs, alpha=0.05, verbose=False):
    n = len(sample)
    sample_values = set(sample)
    theor_values = set(theor_probs.keys())
    
    if not sample_values.issubset(theor_values):
        if verbose:
            print("В выборке присутствуют значения, невозможные при данной гипотезе H0.")
        return float('inf'), 0.0, True

    # Подсчет наблюдаемых частот через стандартный словарь
    counts = {k: 0 for k in theor_probs.keys()}
    for val in sample:
        if val in counts:
            counts[val] += 1
            
    observed = [counts[k] for k in sorted(theor_probs.keys())]
    expected = [theor_probs[k] * n for k in sorted(theor_probs.keys())]
    
    # Объединение малочисленных частот (E_i < 5)
    obs_pooled = []
    exp_pooled = []
    temp_obs = 0
    temp_exp = 0
    
    for o, e in zip(observed, expected):
        temp_obs += o
        temp_exp += e
        if temp_exp >= 5:
            obs_pooled.append(temp_obs)
            exp_pooled.append(temp_exp)
            temp_obs = 0
            temp_exp = 0
            
    if temp_exp > 0:
        if len(exp_pooled) > 0:
            obs_pooled[-1] += temp_obs
            exp_pooled[-1] += temp_exp
        else:
            obs_pooled.append(temp_obs)
            exp_pooled.append(temp_exp)
            
    # Вычисление статистики Хи-квадрат Пирсона
    chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(obs_pooled, exp_pooled))
    df = len(exp_pooled) - 1
    
    if df <= 0:
        p_value = 0.0
    else:
        # Для расчета p-value используем scipy.stats.chi2 (так как это стандартный статистический распределитель)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
    reject_h0 = p_value < alpha
    
    if verbose:
        print(f"  Статистика X^2: {chi2_stat:.4f}")
        print(f"  Степени свободы: {df}")
        print(f"  p-value: {p_value:.4f}")
        if reject_h0:
            print("  Результат: Гипотеза ОТВЕРГАЕТСЯ")
        else:
            print("  Результат: Гипотеза ПРИНИМАЕТСЯ")
            
    return chi2_stat, p_value, reject_h0


# 4. Моделирование и базовый анализ
print("=== 1. МОДЕЛИРОВАНИЕ И ОЦЕНКА ПАРАМЕТРОВ ===\n")

sample_ber = generate_discrete(get_bernoulli_probs(), N)
sample_bin = generate_discrete(get_binomial_probs(), N)

mean_ber_true = p_ber
var_ber_true = p_ber * (1 - p_ber)

mean_bin_true = m_bin * p_bin
var_bin_true = m_bin * p_bin * (1 - p_bin)

mean_ber_emp = calculate_mean(sample_ber)
var_ber_emp = calculate_variance(sample_ber, mean_ber_emp)

mean_bin_emp = calculate_mean(sample_bin)
var_bin_emp = calculate_variance(sample_bin, mean_bin_emp)

print(f"--- Распределение Бернулли Bi(1, {p_ber}) ---")
print(f"Математическое ожидание: Истинное = {mean_ber_true:.4f}, Оценка = {mean_ber_emp:.4f}, Разница = {abs(mean_ber_true - mean_ber_emp):.4f}")
print(f"Дисперсия:               Истинная = {var_ber_true:.4f}, Оценка = {var_ber_emp:.4f}, Разница = {abs(var_ber_true - var_ber_emp):.4f}\n")

print(f"--- Биномиальное распределение Bi({m_bin}, {p_bin}) ---")
print(f"Математическое ожидание: Истинное = {mean_bin_true:.4f}, Оценка = {mean_bin_emp:.4f}, Разница = {abs(mean_bin_true - mean_bin_emp):.4f}")
print(f"Дисперсия:               Истинная = {var_bin_true:.4f}, Оценка = {var_bin_emp:.4f}, Разница = {abs(var_bin_true - var_bin_emp):.4f}\n")


# 5. Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

def get_frequencies(sample, keys):
    counts = {k: 0 for k in keys}
    for val in sample:
        if val in counts:
            counts[val] += 1
    return [counts[k] / N for k in keys]

ber_probs = get_bernoulli_probs()
x_ber = list(ber_probs.keys())
y_ber_theor = list(ber_probs.values())
y_ber_emp = get_frequencies(sample_ber, x_ber)

axes[0].bar([x - 0.15 for x in x_ber], y_ber_theor, width=0.3, label='Теоретические', color='blue', alpha=0.7)
axes[0].bar([x + 0.15 for x in x_ber], y_ber_emp, width=0.3, label='Эмпирические', color='orange', alpha=0.7)
axes[0].set_xticks(x_ber)
axes[0].set_title(f'Бернулли Bi(1, {p_ber})')
axes[0].legend()

bin_probs = get_binomial_probs()
x_bin = list(bin_probs.keys())
y_bin_theor = list(bin_probs.values())
y_bin_emp = get_frequencies(sample_bin, x_bin)

axes[1].bar([x - 0.15 for x in x_bin], y_bin_theor, width=0.3, label='Теоретические', color='blue', alpha=0.7)
axes[1].bar([x + 0.15 for x in x_bin], y_bin_emp, width=0.3, label='Эмпирические', color='orange', alpha=0.7)
axes[1].set_xticks(x_bin)
axes[1].set_title(f'Биномиальное Bi({m_bin}, {p_bin})')
axes[1].legend()

plt.tight_layout()
plt.show()

# 6. Кросс-проверка критерием Хи-квадрат
print("=== 2. ПРОВЕРКА КРИТЕРИЕМ ХИ-КВАДРАТ ===\n")

print("Кросс-проверка 1: Выборка Бернулли проверяется на гипотезу H0 = [Распределение Бернулли]")
pearson_chi2_test(sample_ber, get_bernoulli_probs(), alpha=ALPHA, verbose=True)
print("-" * 50)

print("Кросс-проверка 2: Выборка Биномиальная проверяется на гипотезу H0 = [Биномиальное распределение]")
pearson_chi2_test(sample_bin, get_binomial_probs(), alpha=ALPHA, verbose=True)
print("-" * 50)

print("Кросс-проверка 3: Выборка Биномиальная проверяется на гипотезу H0 = [Распределение Бернулли]")
pearson_chi2_test(sample_bin, get_bernoulli_probs(), alpha=ALPHA, verbose=True)
print("-" * 50)

print("Кросс-проверка 4: Выборка Бернулли проверяется на гипотезу H0 = [Биномиальное распределение]")
pearson_chi2_test(sample_ber, get_binomial_probs(), alpha=ALPHA, verbose=True)
print("\n")


# 7. Оценка ошибки I рода
print("=== 3. ИССЛЕДОВАНИЕ ОШИБКИ I РОДА ===\n")
print(f"Уровень значимости (теоретическая ошибка I рода): {ALPHA}")
print("Выполняем 10 000 симуляций для точной оценки...\n")

M_simulations = 10000
reject_bernoulli = 0
reject_binomial = 0

for _ in range(M_simulations):
    sim_ber = generate_discrete(get_bernoulli_probs(), N)
    sim_bin = generate_discrete(get_binomial_probs(), N)
    
    _, _, rej_ber = pearson_chi2_test(sim_ber, get_bernoulli_probs(), alpha=ALPHA, verbose=False)
    _, _, rej_bin = pearson_chi2_test(sim_bin, get_binomial_probs(), alpha=ALPHA, verbose=False)
    
    if rej_ber: reject_bernoulli += 1
    if rej_bin: reject_binomial += 1

emp_alpha_ber = reject_bernoulli / M_simulations
emp_alpha_bin = reject_binomial / M_simulations

print(f"Эмпирическая ошибка I рода для Бернулли:      {emp_alpha_ber:.4f}")
print(f"Эмпирическая ошибка I рода для Биномиального: {emp_alpha_bin:.4f}")
print("Вывод: эмпирические значения на большом числе опытов стремятся к заданному ε = 0.05.")