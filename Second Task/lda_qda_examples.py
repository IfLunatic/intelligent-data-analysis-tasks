"""
Один файл з прикладами до запитань 1–7.
Запуск: python lda_qda_examples_full.py
Рекомендовані пакети: numpy, pandas, matplotlib, scikit-learn, statsmodels, scipy
"""

import sys
import importlib
import numpy as np

REQUIREMENTS = ["numpy", "pandas", "matplotlib", "scikit-learn", "statsmodels", "scipy"]

def check_requirements():
    missing = []
    for pkg in REQUIREMENTS:
        try:
            importlib.import_module(pkg if pkg != "scikit-learn" else "sklearn")
        except Exception:
            missing.append(pkg)
    if missing:
        print("[INFO] Відсутні пакети:", ", ".join(missing))
        print("Встановіть їх командою:\n  pip install " + " ".join(missing))
    else:
        print("[INFO] Усі необхідні пакети встановлено.")

# -----------------------------
# Q1. Ознаки часових рядів + стаціонарність
# -----------------------------
def q1_timeseries_demo(show_plot: bool=False):
    try:
        from statsmodels.tsa.stattools import adfuller
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[Q1] Потрібен пакет statsmodels та matplotlib:", e)
        return

    np.random.seed(0)
    n = 300
    trend = np.linspace(0, 5, n)
    season = 2*np.sin(2*np.pi*np.arange(n)/12)
    noise = np.random.normal(scale=1.0, size=n)
    y = trend + season + noise
    adf_stat, pvalue, *_ = adfuller(y)
    print(f"[Q1] ADF stat={adf_stat:.3f}, p-value={pvalue:.4f} (p<0.05 -> стаціонарний)")
    if show_plot:
        plt.figure()
        plt.plot(y)
        plt.title("Часовий ряд із трендом і сезонністю")
        plt.xlabel("t"); plt.ylabel("y")
        plt.show()

# -----------------------------
# Q2. Коваріаційна матриця
# -----------------------------
def q2_covariance_demo():
    import numpy as np
    X = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0],
                  [3.0, 4.0, 5.0],
                  [4.0, 3.0, 2.0]])
    S = np.cov(X, rowvar=False, bias=False)  # p×p
    print("[Q2] Коваріаційна матриця S:\n", np.round(S, 3))
    print("[Q2] Симетрична?", np.allclose(S, S.T))

# -----------------------------
# Q3. Лінійний дискримінантний аналіз (LDA)
# -----------------------------
def q3_lda_demo():
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        print("[Q3] Потрібен пакет scikit-learn:", e)
        return

    X, y = make_classification(n_samples=600, n_features=6, n_informative=3,
                               n_redundant=0, n_classes=3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xtr, ytr)
    yp = lda.predict(Xte)
    print("[Q3] LDA classification report:\n", classification_report(yte, yp))
    print("[Q3] Confusion matrix:\n", confusion_matrix(yte, yp))

# -----------------------------
# Q4. Відмінність LDA vs QDA (порівняння)
# -----------------------------
def q4_compare_lda_qda(show_plot: bool=False):
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.datasets import make_blobs
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("[Q4] Потрібен пакет scikit-learn (і matplotlib для графіку):", e)
        return

    X, y = make_blobs(n_samples=500, centers=[[0,0],[3,1]], cluster_std=[1.0, 2.0], random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=7, stratify=y)
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis().fit(Xtr, ytr)
    print(f"[Q4] LDA acc={accuracy_score(yte, lda.predict(Xte)):.3f}")
    print(f"[Q4] QDA acc={accuracy_score(yte, qda.predict(Xte)):.3f}")

    if show_plot:
        xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                             np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z_lda = lda.predict(grid).reshape(xx.shape)
        Z_qda = qda.predict(grid).reshape(xx.shape)
        plt.figure(); plt.contourf(xx, yy, Z_lda, alpha=0.2)
        for cls in np.unique(y): plt.scatter(X[y==cls,0], X[y==cls,1], label=f"class {cls}")
        plt.title("LDA рішення"); plt.legend()
        plt.figure(); plt.contourf(xx, yy, Z_qda, alpha=0.2)
        for cls in np.unique(y): plt.scatter(X[y==cls,0], X[y==cls,1], label=f"class {cls}")
        plt.title("QDA рішення"); plt.legend()
        plt.show()

# -----------------------------
# Q5. Аналіз можливостей LDA/QDA (регуляризація, CV)
# -----------------------------
def q5_cv_shrinkage_lda():
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.datasets import make_classification
        import numpy as np
    except Exception as e:
        print("[Q5] Потрібен пакет scikit-learn:", e)
        return

    X, y = make_classification(n_samples=800, n_features=20, n_informative=6, n_redundant=2,
                               n_classes=3, class_sep=1.2, random_state=0)
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(f"[Q5] CV mean acc={np.mean(scores):.3f}, std={np.std(scores):.3f}")

# -----------------------------
# Q6. Аналоги можливостей Statistics Toolbox у Python
# -----------------------------
def q6_stats_toolbox_equivalents():
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        import statsmodels.formula.api as smf
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.datasets import make_blobs
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:
        print("[Q6] Потрібні пакети scipy, statsmodels, scikit-learn, pandas:", e)
        return

    # 6.1 Описова статистика
    x = np.random.normal(loc=5, scale=2, size=200)
    print("[Q6] Описова статистика: mean=", round(np.mean(x),3), "std=", round(np.std(x, ddof=1),3))

    # 6.2 Підбір розподілу і KS-тест
    mu, sigma = stats.norm.fit(x)
    ks_stat, ks_p = stats.kstest(x, 'norm', args=(mu, sigma))
    print(f"[Q6] Norm fit: mu={mu:.3f}, sigma={sigma:.3f}; KS p={ks_p:.3f}")

    # 6.3 Лінійна регресія (OLS)
    df = pd.DataFrame({'y': x + np.random.normal(scale=1.0, size=len(x)),
                       'x1': x, 'x2': np.random.uniform(size=len(x))})
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    print("[Q6] OLS R^2:", round(model.rsquared, 3))

    # 6.4 Однофакторна ANOVA
    g1 = np.random.normal(0, 1, 30)
    g2 = np.random.normal(0.5, 1, 30)
    g3 = np.random.normal(1.0, 1, 30)
    f_stat, p_val = stats.f_oneway(g1, g2, g3)
    print(f"[Q6] ANOVA: F={f_stat:.3f}, p={p_val:.4f}")

    # 6.5 Кластеризація (k-means)
    Xc, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xc)
    print("[Q6] KMeans inertia:", round(km.inertia_, 2))

    # 6.6 Зниження розмірності (PCA)
    Xp = np.random.normal(size=(200, 5))
    pca = PCA(n_components=2).fit(Xp)
    print("[Q6] PCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))

    # 6.7 Часові ряди (ARIMA)
    ts = np.cumsum(np.random.normal(size=300))  # інтегрований шум
    arima = ARIMA(ts, order=(1,1,1)).fit()
    print("[Q6] ARIMA(1,1,1) AIC:", round(arima.aic, 2))

# -----------------------------
# Q7. Аналоги gscatter та classify у Python
# -----------------------------
def q7_gscatter_and_classify_analogs(show_plot: bool=False):
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[Q7] Потрібні scikit-learn та matplotlib:", e)
        return

    np.random.seed(1)
    X0 = np.random.normal(loc=[0,0], scale=1.0, size=(60,2))
    X1 = np.random.normal(loc=[3,2], scale=1.2, size=(60,2))
    X = np.vstack([X0, X1])
    y = np.array([0]*60 + [1]*60)

    # gscatter-аналог
    if show_plot:
        for cls in np.unique(y):
            pts = X[y==cls]
            plt.scatter(pts[:,0], pts[:,1], label=f"class {cls}")
        plt.legend()
        plt.title("Групована діаграма (аналог gscatter)")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.show()

    # classify-аналог через LDA
    lda = LinearDiscriminantAnalysis().fit(X, y)
    pred = lda.predict([[1,1], [4,3]])
    print("[Q7] Прогнози LDA для [1,1] та [4,3]:", pred)

def run_all():
    print("=== Перевірка залежностей ===")
    check_requirements()
    print("\n=== Q1 ==="); q1_timeseries_demo(show_plot=False)
    print("\n=== Q2 ==="); q2_covariance_demo()
    print("\n=== Q3 ==="); q3_lda_demo()
    print("\n=== Q4 ==="); q4_compare_lda_qda(show_plot=False)
    print("\n=== Q5 ==="); q5_cv_shrinkage_lda()
    print("\n=== Q6 ==="); q6_stats_toolbox_equivalents()
    print("\n=== Q7 ==="); q7_gscatter_and_classify_analogs(show_plot=False)

if __name__ == "__main__":
    # Запустити всі демо якщо не передано аргументів
    if len(sys.argv) == 1:
        run_all()
    else:
        # Можна викликати конкретну функцію, наприклад:
        # python lda_qda_examples_full.py q3_lda_demo
        globals().get(sys.argv[1], run_all)()
