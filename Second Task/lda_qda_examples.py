"""
Один файл з прикладами до запитань 1, 3, 4, 5, 7.
Запуск: python lda_qda_examples.py
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def q1_stationarity_demo():
    """Перевірка стаціонарності (тільки приклад побудови ряду)."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as e:
        print("Встановіть statsmodels для ADF-тесту:", e)
        return
    np.random.seed(0)
    n = 300
    trend = np.linspace(0, 5, n)
    season = 2*np.sin(2*np.pi*np.arange(n)/12)
    noise = np.random.normal(scale=1.0, size=n)
    y = trend + season + noise
    adf_stat, pvalue, *_ = adfuller(y)
    print(f"[Q1] ADF stat={adf_stat:.3f}, p-value={pvalue:.4f}")
    # Візуалізація (за бажанням)
    plt.figure()
    plt.plot(y)
    plt.title("Часовий ряд із трендом і сезонністю")
    plt.xlabel("t"); plt.ylabel("y")
    # plt.show()

def q3_lda_demo():
    """LDA класифікація з макетом набору даних."""
    X, y = make_classification(n_samples=600, n_features=6, n_informative=3,
                               n_redundant=0, n_classes=3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xtr, ytr)
    yp = lda.predict(Xte)
    print("[Q3] LDA classification report:")
    print(classification_report(yte, yp))
    print("[Q3] Confusion matrix:")
    print(confusion_matrix(yte, yp))

def q4_compare_lda_qda():
    """Порівняння LDA vs QDA на простому 2D прикладі."""
    X, y = make_blobs(n_samples=500, centers=[[0,0],[3,1]], cluster_std=[1.0, 2.0], random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=7, stratify=y)
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis().fit(Xtr, ytr)
    print(f"[Q4] LDA acc={accuracy_score(yte, lda.predict(Xte)):.3f}")
    print(f"[Q4] QDA acc={accuracy_score(yte, qda.predict(Xte)):.3f}")
    # Малюнок меж рішень (за бажанням)
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
    # plt.show()

def q5_cv_shrinkage_lda():
    """Перехресна перевірка та shrinkage для LDA."""
    X, y = make_classification(n_samples=800, n_features=20, n_informative=6, n_redundant=2,
                               n_classes=3, class_sep=1.2, random_state=0)
    pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(f"[Q5] CV mean acc={np.mean(scores):.3f}, std={np.std(scores):.3f}")

def q7_gscatter_and_classify_analogs():
    """Приклади Python-аналогів gscatter та classify."""
    np.random.seed(1)
    X0 = np.random.normal(loc=[0,0], scale=1.0, size=(60,2))
    X1 = np.random.normal(loc=[3,2], scale=1.2, size=(60,2))
    X = np.vstack([X0, X1])
    y = np.array([0]*60 + [1]*60)

    # gscatter-аналог
    for cls in np.unique(y):
        pts = X[y==cls]
        plt.scatter(pts[:,0], pts[:,1], label=f"class {cls}")
    plt.legend()
    plt.title("Групована діаграма (аналог gscatter)")
    plt.xlabel("x1"); plt.ylabel("x2")
    # plt.show()

    # classify-аналог через LDA
    lda = LinearDiscriminantAnalysis().fit(X, y)
    pred = lda.predict([[1,1], [4,3]])
    print("[Q7] Прогнози LDA для [1,1] та [4,3]:", pred)

if __name__ == "__main__":
    q1_stationarity_demo()
    q3_lda_demo()
    q4_compare_lda_qda()
    q5_cv_shrinkage_lda()
    q7_gscatter_and_classify_analogs()
