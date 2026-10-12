import math
import matplotlib.pyplot as plt
from scipy import stats

# Количество реализаций (объем выборки)
N = 1000
# Уровень значимости для критерия Пирсона
ALPHA = 0.05

# Параметр распределения Бернулли
p_ber = 0.7

# Параметры биномиального распределения
m_bin = 5
p_bin = 0.25

# Класс линейного конгруэнтного генератора (ЛКГ) для псевдослучайных чисел
class LCG:
    def __init__(self, seed=42):
        # Инициализация параметров генератора: состояние, множитель, приращение, модуль
        self.state = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32

    def random(self):
        # Обновление состояния формулой ЛКГ и возврат значения в диапазоне [0, 1)
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

# Создание глобального объекта генератора с фиксированным seed
lcg = LCG(seed=12345)

def uniform_random_sample(size):
    return [lcg.random() for _ in range(size)]


# Функция формирования теоретических вероятностей Бернулли
def get_bernoulli_probs():
    return {0: 1 - p_ber, 1: p_ber}

# Функция формирования теоретических вероятностей биномиального закона
def get_binomial_probs():
    probs = {}
    for k in range(m_bin + 1):
        # Вычисление числа сочетаний через встроенный модуль math.comb
        comb = math.comb(m_bin, k)
        probs[k] = comb * (p_bin ** k) * ((1 - p_bin) ** (m_bin - k))
    return probs

# Функция моделирования дискретной случайной величины методом обратной функции
def generate_discrete(probs_dict, size):
    values = list(probs_dict.keys())
    probabilities = list(probs_dict.values())
    
    # Построение массива кумулятивных вероятностей (функции распределения)
    cdf = []
    accumulated = 0.0
    for p in probabilities:
        accumulated += p
        cdf.append(accumulated)
    # Принудительная установка последнего элемента равным 1.0 для избежания погрешностей
    cdf[-1] = 1.0
    
    # Получение равномерно распределенных случайных чисел
    uniforms = uniform_random_sample(size)
    sample = []
    
    # Преобразование равномерных чисел в дискретную СВ методом инверсии
    for u in uniforms:
        chosen_val = values[-1]
        for idx, bound in enumerate(cdf):
            # Проверка попадания случайного числа в интервал кумулятивной вероятности
            if u < bound:
                chosen_val = values[idx]
                break
        sample.append(chosen_val)
        
    return sample


# Функция вычисления математического ожидания выборки
def calculate_mean(sample):
    return sum(sample) / len(sample)

# Функция вычисления несмещенной оценки дисперсии
def calculate_variance(sample, mean_val):
    n = len(sample)
    # Если в выборке меньше или равно 1 элемента, дисперсия равна нулю
    if n <= 1:
        return 0.0
    # Сумма квадратов отклонений элементов от выборочного среднего
    sq_diff_sum = sum((x - mean_val) ** 2 for x in sample)
    return sq_diff_sum / (n - 1)


# Функция проверки гипотезы с помощью критерия согласия Пирсона (Хи-квадрат)
def pearson_chi2_test(sample, theor_probs, alpha=0.05, verbose=False):
    n = len(sample)
    sample_values = set(sample)
    theor_values = set(theor_probs.keys())
    
    # Проверка на наличие в выборке значений, невозможных по теоретическому закону
    if not sample_values.issubset(theor_values):
        if verbose:
            print("В выборке присутствуют значения, невозможные при данной гипотезе H0.")
        return float('inf'), 0.0, True

    # Подсчет наблюдаемых частот каждого значения в выборке
    counts = {k: 0 for k in theor_probs.keys()}
    for val in sample:
        if val in counts:
            counts[val] += 1
            
    # Формирование упорядоченных массивов наблюдаемых и ожидаемых частот
    observed = [counts[k] for k in sorted(theor_probs.keys())]
    expected = [theor_probs[k] * n for k in sorted(theor_probs.keys())]
    
    obs_pooled = []
    exp_pooled = []
    temp_obs = 0
    temp_exp = 0
    
    for o, e in zip(observed, expected):
        temp_obs += o
        temp_exp += e
        # Если накопили сумму ожидаемых частот >= 5, формируем группу
        if temp_exp >= 5:
            obs_pooled.append(temp_obs)
            exp_pooled.append(temp_exp)
            temp_obs = 0
            temp_exp = 0
            
    # Если на последнем шаге осталась группа с малой частотой, присоединяем ее к предыдущей
    if temp_exp > 0:
        if len(exp_pooled) > 0:
            obs_pooled[-1] += temp_obs
            exp_pooled[-1] += temp_exp
        else:
            obs_pooled.append(temp_obs)
            exp_pooled.append(temp_exp)
            
    # Вычисление значения статистики хи-квадрат Пирсона
    chi2_stat = sum(((o - e) ** 2) / e for o, e in zip(obs_pooled, exp_pooled))
    # Определение числа степеней свободы (количество групп минус 1)
    df = len(exp_pooled) - 1
    
    if df <= 0:
        p_value = 0.0
    else:
        # Расчет p-value на основе функции распределения хи-квадрат
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
    # Принятие или отклонение нулевой гипотезы на основе уровня значимости
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


# Блок моделирования и первичного анализа выборок
print("1. МОДЕЛИРОВАНИЕ И ОЦЕНКА ПАРАМЕТРОВ\n")

# Генерация выборок для обоих законов распределения
sample_ber = generate_discrete(get_bernoulli_probs(), N)
sample_bin = generate_discrete(get_binomial_probs(), N)

# Расчет истинных теоретических характеристик
mean_ber_true = p_ber
var_ber_true = p_ber * (1 - p_ber)

mean_bin_true = m_bin * p_bin
var_bin_true = m_bin * p_bin * (1 - p_bin)

# Расчет эмпирических оценок по полученным выборкам
mean_ber_emp = calculate_mean(sample_ber)
var_ber_emp = calculate_variance(sample_ber, mean_ber_emp)

mean_bin_emp = calculate_mean(sample_bin)
var_bin_emp = calculate_variance(sample_bin, mean_bin_emp)

# Вывод результатов сравнения для распределения Бернулли
print(f"Распределение Бернулли Bi(1, {p_ber})")
print(f"Математическое ожидание: Истинное = {mean_ber_true:.4f}, Оценка = {mean_ber_emp:.4f}, Разница = {abs(mean_ber_true - mean_ber_emp):.4f}")
print(f"Дисперсия:               Истинная = {var_ber_true:.4f}, Оценка = {var_ber_emp:.4f}, Разница = {abs(var_ber_true - var_ber_emp):.4f}\n")

# Вывод результатов сравнения для биномиального распределения
print(f"Биномиальное распределение Bi({m_bin}, {p_bin})")
print(f"Математическое ожидание: Истинное = {mean_bin_true:.4f}, Оценка = {mean_bin_emp:.4f}, Разница = {abs(mean_bin_true - mean_bin_emp):.4f}")
print(f"Дисперсия:               Истинная = {var_bin_true:.4f}, Оценка = {var_bin_emp:.4f}, Разница = {abs(var_bin_true - var_bin_emp):.4f}\n")


# Блок визуализации данных
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

def get_frequencies(sample, keys):
    counts = {k: 0 for k in keys}
    for val in sample:
        if val in counts:
            counts[val] += 1
    return [counts[k] / N for k in keys]

# Подготовка данных для гистограммы распределения Бернулли
ber_probs = get_bernoulli_probs()
x_ber = list(ber_probs.keys())
y_ber_theor = list(ber_probs.values())
y_ber_emp = get_frequencies(sample_ber, x_ber)

# Построение графика для Бернулли
axes[0].bar([x - 0.15 for x in x_ber], y_ber_theor, width=0.3, label='Теоретические', color='blue', alpha=0.7)
axes[0].bar([x + 0.15 for x in x_ber], y_ber_emp, width=0.3, label='Эмпирические', color='orange', alpha=0.7)
axes[0].set_xticks(x_ber)
axes[0].set_title(f'Бернулли Bi(1, {p_ber})')
axes[0].legend()

# Подготовка данных для гистограммы биномиального распределения
bin_probs = get_binomial_probs()
x_bin = list(bin_probs.keys())
y_bin_theor = list(bin_probs.values())
y_bin_emp = get_frequencies(sample_bin, x_bin)

# Построение графика для биномиального распределения
axes[1].bar([x - 0.15 for x in x_bin], y_bin_theor, width=0.3, label='Теоретические', color='blue', alpha=0.7)
axes[1].bar([x + 0.15 for x in x_bin], y_bin_emp, width=0.3, label='Эмпирические', color='orange', alpha=0.7)
axes[1].set_xticks(x_bin)
axes[1].set_title(f'Биномиальное Bi({m_bin}, {p_bin})')
axes[1].legend()

plt.tight_layout()
plt.show()

# Блок проверки критерием хи-квадрат (кросс-проверка)
print("2. ПРОВЕРКА КРИТЕРИЕМ ХИ-КВАДРАТ\n")

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


# Блок исследования вероятности ошибки I рода
print("3. ИССЛЕДОВАНИЕ ОШИБКИ I РОДА\n")
print(f"Уровень значимости (теоретическая ошибка I рода): {ALPHA}")
print("Выполняем 10 000 симуляций для точной оценки...\n")

M_simulations = 10000
reject_bernoulli = 0
reject_binomial = 0

# Цикл многократной симуляции для оценки эмпирического уровня ошибки
for _ in range(M_simulations):
    sim_ber = generate_discrete(get_bernoulli_probs(), N)
    sim_bin = generate_discrete(get_binomial_probs(), N)
    
    # Тестирование сгенерированных выборок на их истинных законах распределения
    _, _, rej_ber = pearson_chi2_test(sim_ber, get_bernoulli_probs(), alpha=ALPHA, verbose=False)
    _, _, rej_bin = pearson_chi2_test(sim_bin, get_binomial_probs(), alpha=ALPHA, verbose=False)
    
    # Подсчет случаев ошибочного отклонения верной гипотезы
    if rej_ber: reject_bernoulli += 1
    if rej_bin: reject_binomial += 1

# Вычисление эмпирической частоты ошибки I рода
emp_alpha_ber = reject_bernoulli / M_simulations
emp_alpha_bin = reject_binomial / M_simulations

print(f"Эмпирическая ошибка I рода для Бернулли:      {emp_alpha_ber:.4f}")
print(f"Эмпирическая ошибка I рода для Биномиального: {emp_alpha_bin:.4f}")
print("Вывод: эмпирические значения на большом числе опытов стремятся к заданному ε = 0.05.")