# Запуск: python ida_task_code.py
# Вимоги: pip install numpy pandas scipy matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Допоміжні функції ----------

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

def normality_test(vec):
    s = pd.Series(vec).dropna().values
    n = len(s)
    if n == 0:
        return False, np.nan, "no-data"
    if n > 500:
        mu = np.mean(s)
        sigma = np.std(s, ddof=1) if n > 1 else 1.0
        stat, p = stats.kstest(s, 'norm', args=(mu, sigma))
        return p > 0.05, p, "kolmogorov-smirnov"
    else:
        stat, p = stats.shapiro(s)
        return p > 0.05, p, "shapiro-wilk"

def choose_and_compute_corr(x, y):
    x_norm, x_p, x_test = normality_test(x)
    y_norm, y_p, y_test = normality_test(y)
    if x_norm and y_norm:
        coef, pval = stats.pearsonr(x, y)
        method = "Pearson"
    else:
        coef, pval = stats.spearmanr(x, y)
        method = "Spearman"
    return method, coef, pval, x_test, x_p, y_test, y_p

def print_corr_summary(title, method, coef, pval, x_test, x_p, y_test, y_p):
    print("\n" + "="*100)
    print(title)
    print("-"*100)
    print(f"[Нормальність] x: {x_test}, p={x_p:.4g} | y: {y_test}, p={y_p:.4g}")
    print(f"[Кореляція]   Метод: {method} | коефіцієнт: {coef:.3f} | p-value: {pval:.4g}")
    print(f"[Інтерпретація сили] {strength_label(coef)}")
    if pval < 0.05:
        print("Висновок: зв’язок статистично значущий (відхиляємо H0: ρ=0).")
    else:
        print("Висновок: зв’язок НЕ значущий (не відхиляємо H0: ρ=0).")
    print("="*100 + "\n")

# ---------- Приклади ----------

def example_1_linear():
    print("# Перший приклад — лінійний зв’язок (~нормальні дані)")
    np.random.seed(0)
    x = np.random.normal(0, 1, 200)
    y = 0.6*x + np.random.normal(0, 1, 200)
    method, coef, pval, x_test, x_p, y_test, y_p = choose_and_compute_corr(x, y)
    print_corr_summary("Лінійний кейс (очікуємо Pearson)", method, coef, pval, x_test, x_p, y_test, y_p)
    plt.figure(); plt.scatter(x, y); plt.title("Приклад 1: Лінійний"); plt.grid(True, alpha=0.3); plt.show()

def example_2_monotonic():
    print("# Другий приклад — монотонний, але не лінійний зв’язок")
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 250)
    y = np.exp(x) + rng.normal(0, 0.4, 250)
    method, coef, pval, x_test, x_p, y_test, y_p = choose_and_compute_corr(x, y)
    print_corr_summary("Монотонна нелінійність (очікуємо Spearman)", method, coef, pval, x_test, x_p, y_test, y_p)
    plt.figure(); plt.scatter(x, y); plt.title("Приклад 2: Монотонна нелінійність"); plt.grid(True, alpha=0.3); plt.show()

def example_3_kendall():
    print("# Третій приклад — Kendall tau-b (невеликі вибірки/зв’язані ранги)")
    x = np.array([1,2,2,3,4,4,5,5])
    y = np.array([3,1,2,2,5,4,4,6])
    tau, p = stats.kendalltau(x, y)
    print_corr_summary("Kendall tau-b", "Kendall tau-b", tau, p, "—", float('nan'), "—", float('nan'))
    plt.figure(); plt.scatter(x, y); plt.title("Приклад 3: Kendall tau-b"); plt.grid(True, alpha=0.3); plt.show()

def example_4_csv(csv_path="lipoprotein_hemoglobin.csv", x_col="lipoproteins", y_col="hemoglobin"):
    print("# Четвертий приклад — CSV (ліпопротеїни vs гемоглобін)")
    if not os.path.exists(csv_path):
        print(f"Не знайдено {csv_path}. Створю демо-таблицю з синтетичними даними…")
        np.random.seed(123)
        n = 60
        lipoproteins = np.random.normal(3.2, 0.8, n).clip(0.5, None)
        hemoglobin   = (14.8 - 0.25*lipoproteins) + np.random.normal(0, 0.6, n)
        df_demo = pd.DataFrame({x_col: lipoproteins.round(3), y_col: hemoglobin.round(3)})
        df_demo.to_csv(csv_path, index=False)
        print(f"Збережено демо CSV: {csv_path} (колонки: {x_col}, {y_col})")
    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df[x_col], errors='coerce').dropna().values
    y = pd.to_numeric(df[y_col], errors='coerce').dropna().values
    rho, p = stats.spearmanr(x, y, nan_policy='omit')
    print(f"Spearman rho = {rho:.3f}, p = {p:.3g} | {strength_label(rho)} зв’язок")
    plt.figure(); plt.scatter(x, y, s=24); plt.title(f"Ліпопротеїни vs Гемоглобін (ρ={rho:.3f}, p={p:.3g})"); plt.grid(True, alpha=0.3); plt.show()

def example_5_corr_matrix(method='spearman'):
    print("# П’ятий приклад — матриця кореляцій кількох змінних")
    rng = np.random.default_rng(7)
    n = 300
    a = rng.normal(0, 1, n)
    b = 0.5*a + rng.normal(0, 1, n)
    c = np.exp(0.7*a) + rng.normal(0, 0.5, n)
    d = rng.normal(0, 1, n)
    df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d})
    C = df.corr(method=method)
    print(f"Матриця кореляцій ({method}):\n", C.round(3))

# ---------- main ----------

def main():
    example_1_linear()
    example_2_monotonic()
    example_3_kendall()
    example_4_csv()       
    example_5_corr_matrix()

if __name__ == "__main__":
    main()
