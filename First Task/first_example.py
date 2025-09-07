# Запуск: python iad_correlation_examples.py
# Вимоги: pip install numpy pandas scipy matplotlib


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def strength_label(r: float) -> str:
    bins = [ -1.00, -0.75, -0.50, -0.25, -0.00, 0.00, 0.25, 0.50, 0.75, 1.00 ]
    labels = [
        "дуже високий негативний", "високий негативний", "середній негативний",
        "слабкий негативний", "слабкий позитивний", "слабкий позитивний",
        "середній позитивний", "високий позитивний", "дуже високий позитивний"
    ]
    for (lo, hi), lab in zip(zip(bins[:-1], bins[1:]), labels):
        if (r >= lo) and (r <= hi):
            return lab
    return "невизначено"

def choose_and_compute_corr(x, y):
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    shapiro_x = stats.shapiro(x)
    shapiro_y = stats.shapiro(y)
    is_norm_x = shapiro_x.pvalue > 0.05
    is_norm_y = shapiro_y.pvalue > 0.05
    if is_norm_x and is_norm_y:
        coef, pval = stats.pearsonr(x, y)
        method = 'Pearson'
    else:
        coef, pval = stats.spearmanr(x, y)
        method = 'Spearman'
    return method, coef, pval, shapiro_x.pvalue, shapiro_y.pvalue

def print_corr_summary(title, method, coef, pval):
    print("\n" + "="*80)
    print(title)
    print("-"*80)
    print(f"Метод: {method}")
    print(f"Коефіцієнт: {coef:.3f}")
    print(f"p-value: {pval:.4g}")
    print(f"Сила/тип зв'язку: {strength_label(coef)}")
    if pval < 0.05:
        print("Висновок: зв’язок статистично значущий (відхиляємо H0: ρ=0).")
    else:
        print("Висновок: зв’язок НЕ значущий (не відхиляємо H0: ρ=0).")
    print("="*80 + "\n")

# Перший приклад
def first_example():
    print("# Перший приклад")
    np.random.seed(0)
    x = np.random.normal(0, 1, 200)
    y = 0.6*x + np.random.normal(0, 1, 200)
    method, coef, pval, p_x, p_y = choose_and_compute_corr(x, y)
    print_corr_summary("Лінійний кейс (очікуємо Pearson)", method, coef, pval)
    plt.scatter(x, y)
    plt.title('Перший приклад')
    plt.show()

# Другий приклад
def second_example():
    print("# Другий приклад")
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 250)
    y = np.exp(x) + rng.normal(0, 0.4, 250)
    method, coef, pval, p_x, p_y = choose_and_compute_corr(x, y)
    print_corr_summary("Нелінійно-монотонний кейс (очікуємо Spearman)", method, coef, pval)
    plt.scatter(x, y)
    plt.title('Другий приклад')
    plt.show()

# Третій приклад
def third_example():
    print("# Третій приклад")
    x = np.array([1,2,2,3,4,4,5,5])
    y = np.array([3,1,2,2,5,4,4,6])
    tau, p = stats.kendalltau(x, y)
    print_corr_summary("Kendall tau-b", "Kendall tau-b", tau, p)
    plt.scatter(x, y)
    plt.title('Третій приклад')
    plt.show()

# Четвертий приклад (CSV)
def fourth_example(csv_path="your_data.csv"):
    print("# Четвертий приклад")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        x, y = df['x'].values, df['y'].values
    else:
        np.random.seed(42)
        x = np.random.normal(0,1,120)
        y = 0.4*x + np.random.normal(0,1,120)
    method, coef, pval, p_x, p_y = choose_and_compute_corr(x, y)
    print_corr_summary("Ваші дані / CSV", method, coef, pval)
    plt.scatter(x, y)
    plt.title('Четвертий приклад')
    plt.show()

# П'ятий приклад
def fifth_example():
    print("# П'ятий приклад")
    rng = np.random.default_rng(7)
    a = rng.normal(0,1,300)
    b = 0.5*a + rng.normal(0,1,300)
    c = np.exp(0.7*a) + rng.normal(0,0.5,300)
    d = rng.normal(0,1,300)
    df = pd.DataFrame({'a':a,'b':b,'c':c,'d':d})
    C = df.corr(method='spearman')
    print(C)

def main():
    first_example()
    second_example()
    third_example()
    fourth_example()
    fifth_example()

if __name__ == "__main__":
    main()
