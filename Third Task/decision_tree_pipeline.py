#Завдання 2 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def load_dataset_from_csv(
    path: str,
    target: str,
    auto_encode: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target == "-1":
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    else:
        if target not in df.columns:
            raise ValueError(f"Не знайдено цільовий стовпець '{target}' у CSV.")
        y = df[target]
        X = df.drop(columns=[target])

    # Авто-кастинг категоріальних
    if auto_encode:
        # Позначаємо категоріальні як object/category/boolean
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Повністю імпутимо пропуски: медіана для числових, найчастіше для категоріальних
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]), num_cols),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), cat_cols)
            ],
            remainder="drop"
        )

        X_processed = preprocessor.fit_transform(X)

        # Відновимо імена фіч (для важливостей)
        new_feature_names = []
        if num_cols:
            new_feature_names.extend(num_cols)
        if cat_cols:
            # витягнемо імена після OneHot
            ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            new_feature_names.extend(ohe_feature_names)

        X = pd.DataFrame(X_processed, columns=new_feature_names)
    else:
        # Базова імпутація числових/категоріальних без кодування
        for col in X.columns:
            if X[col].dtype.kind in "biufc":
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].astype(str).fillna("NA")

    return X, y


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    y = y.map(dict(enumerate(iris.target_names)))  # зробимо зрозумілі ярлики
    return X, y


def choose_alpha_ccp(X_train, y_train, X_val, y_val, random_state=42) -> float:
    """
    Перебір alpha з cost-complexity pruning path і вибір за кращою валідаційною accuracy.
    """
    tmp_tree = DecisionTreeClassifier(random_state=random_state)
    tmp_tree.fit(X_train, y_train)
    path = tmp_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    # Уникнемо нульового дерева: пропустимо найбільше alpha, що повністю зрубує дерево
    alphas = alphas[:-1] if len(alphas) > 1 else alphas

    best_alpha = 0.0
    best_acc = -np.inf
    for a in alphas:
        clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=a)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_alpha = a
    return float(best_alpha)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: Optional[int] = None,
    scale_numeric: bool = False,
    do_cv: bool = True
):
    # train/val/test: виділимо валід. шматок для вибору alpha
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_state, stratify=y_train_full
    )

    # Масштабування числових (для дерева не критично, але інколи корисно)
    if scale_numeric:
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Підбір alpha (обрізка)
    best_alpha = choose_alpha_ccp(X_train, y_train, X_val, y_val, random_state=random_state)

    clf = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        ccp_alpha=best_alpha
    )
    clf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Прогнози та метрики
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Крос-валідація (за бажанням)
    cv_mean, cv_std = (np.nan, np.nan)
    if do_cv:
        cv_scores = cross_val_score(clf, X_train_full, y_train_full, cv=5)
        cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))

    # Статистика дерева
    depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()

    return {
        "model": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "depth": int(depth),
        "n_leaves": int(n_leaves),
        "best_alpha": best_alpha
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Справжній клас",
        xlabel="Прогнозований клас",
        title="Матриця неточностей"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # підписи клітинок
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tree_png(model: DecisionTreeClassifier, feature_names: List[str], class_names: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True
    )
    plt.title("Дерево рішень (обрізане)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rules(model: DecisionTreeClassifier, feature_names: List[str], class_names: List[str], out_path: str):
    text = export_text(model, feature_names=feature_names, show_weights=True)
    # Замінемо номери класів на імена (якщо вони є)
    if class_names:
        for idx, name in enumerate(class_names):
            text = text.replace(f"class: {idx}", f"class: {name}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Синтез та аналіз дерева рішень.")
    parser.add_argument("--csv", type=str, default=None, help="Шлях до CSV із даними.")
    parser.add_argument("--target", type=str, default=None,
                        help="Назва цільового стовпця або -1 (останній).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Частка тесту (0..1).")
    parser.add_argument("--max-depth", type=int, default=None, help="Макс. глибина дерева (опціонально).")
    parser.add_argument("--no-cv", action="store_true", help="Вимкнути крос-валідацію.")
    parser.add_argument("--scale", action="store_true", help="Масштабувати числові ознаки.")
    parser.add_argument("--auto-encode", action="store_true", help="One-hot для категоріальних + імпутація.")
    parser.add_argument("--outdir", type=str, default="dt_outputs", help="Каталог для результатів.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Дані
    if args.csv:
        if args.target is None:
            print("⚠️  Будь ласка, вкажіть --target (назва або -1). Наприклад: --target -1")
            sys.exit(1)
        X, y = load_dataset_from_csv(args.csv, args.target, auto_encode=args.auto_encode)
        class_names = sorted([str(c) for c in pd.Series(y).unique()])
        print(f"Завантажено CSV: {args.csv}. Розмір X={X.shape}, унікальних класів={len(class_names)}")
    else:
        X, y = load_iris_dataset()
        class_names = sorted([str(c) for c in pd.Series(y).unique()])
        print(f"Використовую Iris. Розмір X={X.shape}, класи={class_names}")

    # 2) Навчання + оцінювання
    results = train_and_evaluate(
        X, y,
        test_size=args.test_size,
        random_state=42,
        max_depth=args.max_depth,
        scale_numeric=args.scale,
        do_cv=not args.no_cv
    )

    model: DecisionTreeClassifier = results["model"]

    # 3) Вивід метрик у консоль
    print("\n=== ПІДСУМКИ ===")
    print(f"Accuracy (test): {results['accuracy']:.4f}")
    if not args.no_cv:
        print(f"CV Accuracy (mean±std, cv=5): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"Глибина дерева: {results['depth']}")
    print(f"Кількість листків: {results['n_leaves']}")
    print(f"Обраний ccp_alpha: {results['best_alpha']:.6f}")
    print("\nClassification report:")
    print(results["report"])

    # 4) Збереження візуалізацій та правил
    cm_png = os.path.join(args.outdir, "confusion_matrix.png")
    tree_png = os.path.join(args.outdir, "decision_tree.png")
    rules_txt = os.path.join(args.outdir, "rules.txt")
    fi_csv   = os.path.join(args.outdir, "feature_importances.csv")

    labels = class_names if class_names else sorted(pd.Series(results["y_test"]).unique())
    plot_confusion_matrix(results["confusion_matrix"], labels, cm_png)
    plot_tree_png(model, list(X.columns), class_names, tree_png)
    save_rules(model, list(X.columns), class_names, rules_txt)

    # 5) Важливості ознак
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values("importance", ascending=False)
        fi.to_csv(fi_csv, index=False)
        print("\nТоп-10 ознак за важливістю:")
        print(fi.head(10).to_string(index=False))
    else:
        print("\nМодель не надає важливостей ознак.")

    print(f"\nФайли збережено у: {os.path.abspath(args.outdir)}")
    print(f"- Дерево: {tree_png}")
    print(f"- Матриця неточностей: {cm_png}")
    print(f"- Правила: {rules_txt}")
    print(f"- Важливості ознак: {fi_csv}")


if __name__ == "__main__":
    main()
