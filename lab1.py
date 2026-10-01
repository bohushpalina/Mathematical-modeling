# Лабораторная работа №1
# Вариант 1

import math
from math import floor
import matplotlib.pyplot as plt
import numpy as np

# По условию
N = 1000
M = 2 ** 31

# Уровень значимости
EPS = 0.05

# Переменные моего варианта
A0 = 68921
BETA = 68921

# Объем вспомогательной последовательности 
K = 48


def multiplicative_generator(a0, beta, n):
    values = []
    a = a0
    i = 0

    while i < n:
        a = (beta * a) % M
        value = a / M
        values.append(value)
        i += 1

    return values


def maclaren_marsaglia(n, k):
    first = multiplicative_generator(68921, 68921, n + k)
    second = multiplicative_generator(79507, 79507, n + k)

    table = []
    i = 0

    # Первоначальное заполнение таблицы V - из второго массива
    while i < k:
        table.append(second[i])
        i += 1

    result = []
    t = k
    i = 0

    while i < n:
        # взятие значения из вспомогательной таблицы
        value = table[floor(first[t] * k)]
        # добавление в результативный массив
        result.append(value)
        # обновление значения во вспомогательной таблице
        table[floor(first[t] * k)] = second[t]

        t += 1
        i += 1

    return result


def kolmogorov_test(values):
    sorted_values = sorted(values)
    d = 0
    i = 1

    # Максимальное отклонение
    while i <= N:
        x = sorted_values[i - 1]

        d1 = i / N - x
        d2 = x - (i - 1) / N
        if d1 > d:
            d = d1
        if d2 > d:
            d = d2
        i += 1

    sqrt_n = math.sqrt(N)
    statistic = sqrt_n * d
    # Табличное значение для EPS = 0.05
    critical = 1.36

    if statistic < critical:
        result = "H0(Нулевая гипотеза) принимается"
    else:
        result = "H0(Нулевая гипотеза) отвергается"
    return d, statistic, critical, result


def pearson_test(values):
    intervals = 10
    frequencies = []
    i = 0

    while i < intervals:
        frequencies.append(0)
        i += 1

    # Подсчёт частот
    i = 0
    while i < N:
        value = values[i]

        # Определение к какому промежутку принадлежит значение
        interval = floor(value * intervals)

        # Крайний случай
        if interval >= intervals:
            interval = intervals - 1
        frequencies[interval] += 1
        i += 1

    expected = N / intervals
    chi_square = 0

    i = 0
    while i < intervals:
        difference = frequencies[i] - expected
        chi_square = chi_square + (
            difference * difference / expected
        )
        i += 1

    # Табличное значение для EPS = 0.05
    critical = 16.919
    if chi_square < critical:
        result = "H0(Нулевая гипотеза) принимается"
    else:
        result = "H0(Нулевая гипотеза) отвергается"

    return frequencies, chi_square, critical, result


def print_generator_result(name, values):
    print()
    print(name)

    print("Количество реализаций:", N)

    print()
    print("Первые 10 значений:")

    i = 0

    while i < 10:
        print(i + 1, values[i])
        i += 1

    print()
    print("Критерий Колмогорова:")

    d, statistic, critical, result = kolmogorov_test(values)

    print("Dn =", d)
    print("sqrt(n) * Dn =", statistic)
    print("Критическое значение =", critical)
    print("Результат:", result)

    print()
    print("Критерий Пирсона:")

    frequencies, chi_square, critical, result = pearson_test(values)

    print("Частоты по 10 интервалам:")

    i = 0

    while i < 10:
        print(
            "[" + str(i / 10) +
            "; " +
            str((i + 1) / 10) +
            ") : ",
            frequencies[i]
        )

        i += 1

    print()
    print("chi^2 =", chi_square)
    print("Критическое значение =", critical)
    print("Результат:", result)


def plot_combined_graphics(mk_vals, mm_vals):
    """2 гистограммы + 2 диаграммы рассеивания."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(
        mk_vals,
        bins=10,
        range=(0, 1),
        density=True,
        edgecolor='black',
        alpha=0.8
    )

    axes[0, 0].axhline(
        y=1.0,
        linestyle='--',
        linewidth=1.5,
        label='Теоретическая плотность f(x) = 1'
    )

    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_xticks(np.arange(0, 1.1, 0.1))
    axes[0, 0].set_title(
        "МКД: гистограмма плотности",
        fontsize=11,
        fontweight='bold'
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Плотность")
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 0].legend()



    axes[0, 1].hist(
        mm_vals,
        bins=10,
        range=(0, 1),
        density=True,
        edgecolor='black',
        alpha=0.8
    )

    axes[0, 1].axhline(
        y=1.0,
        linestyle='--',
        linewidth=1.5,
        label='Теоретическая плотность f(x) = 1'
    )

    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_xticks(np.arange(0, 1.1, 0.1))
    axes[0, 1].set_title(
        "Макларен-Марсалья: гистограмма плотности",
        fontsize=11,
        fontweight='bold'
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Плотность")
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 1].legend()


    x_mk = np.array(mk_vals[:-1])
    y_mk = np.array(mk_vals[1:])

    axes[1, 0].scatter(
        x_mk,
        y_mk,
        s=12,
        alpha=0.65,
        marker='o',
        linewidths=0
    )

    axes[1, 0].set_title(
        "МКД: диаграмма рассеивания $x_i$ и $x_{i+1}$",
        fontsize=11,
        fontweight='bold'
    )

    axes[1, 0].set_xlabel("$x_i$")
    axes[1, 0].set_ylabel("$x_{i+1}$")

    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 0].set_aspect('equal', adjustable='box')

    axes[1, 0].set_xticks(np.arange(0, 1.1, 0.1))
    axes[1, 0].set_yticks(np.arange(0, 1.1, 0.1))

    axes[1, 0].grid(
        True,
        linestyle=':',
        alpha=0.35
    )


    x_mm = np.array(mm_vals[:-1])
    y_mm = np.array(mm_vals[1:])

    axes[1, 1].scatter(
        x_mm,
        y_mm,
        s=12,
        alpha=0.65,
        marker='o',
        linewidths=0
    )

    axes[1, 1].set_title(
        "Макларен-Марсалья: диаграмма рассеивания $x_i$ и $x_{i+1}$",
        fontsize=11,
        fontweight='bold'
    )

    axes[1, 1].set_xlabel("$x_i$")
    axes[1, 1].set_ylabel("$x_{i+1}$")

    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    axes[1, 1].set_aspect('equal', adjustable='box')

    axes[1, 1].set_xticks(np.arange(0, 1.1, 0.1))
    axes[1, 1].set_yticks(np.arange(0, 1.1, 0.1))

    axes[1, 1].grid(
        True,
        linestyle=':',
        alpha=0.35
    )

    plt.tight_layout()
    plt.show()


mk_generator = multiplicative_generator(A0, BETA, N)
print_generator_result("Мультипликативный конгруэнтный датчик", mk_generator)

mm_generator = maclaren_marsaglia(N, K)
print("-----------------------------")
print_generator_result("Датчик Макларена-Марсальи", mm_generator)

plot_combined_graphics(mk_generator, mm_generator)