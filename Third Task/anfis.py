#Завдання 9

"""
ANFIS (Jang, 1993) — Sugeno 1-го порядку з ґаусовими MF.
- N_f: кількість ознак
- M:   кількість MF на кожну ознаку (загалом правил = M**N_f)
- Передумови (premise): ґаусівські MF з параметрами (c, s)
- Наслідки (consequents): лінійні по входу: z_k(x) = w_k^T x + b_k
- Агрегація: добуток (T-норма), нормування вогнів, зважена сума виходів правил

Навчання (гібридне, K епох):
  1) Фіксуємо MF → розв'язуємо зважену МНК для наслідків (векторно).
  2) Фіксуємо наслідки → оновлюємо (c, s) градієнтом (Adam).

Демо: апроксимація 2D-функції y = sin(x1) + 0.5*cos(x2) + шум.
"""

from dataclasses import dataclass
from itertools import product
import numpy as np

# --------------------- Утиліти ---------------------

def gaussian(x, c, s):
    # уникнемо нулів/поганої масштабу s
    s = np.maximum(s, 1e-6)
    return np.exp(-0.5 * ((x - c) / s) ** 2)

def make_grid_centers_stds(X, M):
    """
    Обчислити стартові (c, s) для кожної ознаки:
    - центри c: рівновіддалені по діапазону
    - s: однакові, ~ відстань між сусідніми центрами / sqrt(2)
    Повертає:
      centers: list length N_f, кожен — (M,) масив
      stds:    list length N_f, кожен — (M,) масив
    """
    N, N_f = X.shape
    centers, stds = [], []
    for j in range(N_f):
        lo, hi = np.min(X[:, j]), np.max(X[:, j])
        c = np.linspace(lo, hi, M)
        if M > 1:
            d = (hi - lo) / (M - 1)
            s = np.full(M, d / np.sqrt(2) + 1e-6)
        else:
            s = np.full(M, (hi - lo + 1e-6))
        centers.append(c)
        stds.append(s)
    return centers, stds

# --------------------- ANFIS модель ---------------------

@dataclass
class ANFISConfig:
    M: int = 3                  # MF на ознаку
    lr: float = 0.01            # навчальна швидкість для (c,s)
    epochs: int = 100
    batch_size: int = 64
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    random_state: int = 42
    verbose: bool = True

class ANFIS:
    def __init__(self, config: ANFISConfig):
        self.cfg = config
        self.is_fitted = False

    def _init_params(self, X):
        rng = np.random.default_rng(self.cfg.random_state)
        self.N, self.N_f = X.shape
        self.centers, self.stds = make_grid_centers_stds(X, self.cfg.M)

        # Всі комбінації MF по ознаках = правила
        self.rule_indices = np.array(list(product(*[range(self.cfg.M) for _ in range(self.N_f)])))
        self.R = self.rule_indices.shape[0]  # кількість правил

        # Наслідкові параметри: для кожного правила w \in R^{N_f}, b \in R
        self.W = rng.normal(scale=0.1, size=(self.R, self.N_f))
        self.b = rng.normal(scale=0.1, size=(self.R,))

        # Adam стани для (c,s)
        self.m_c = [np.zeros_like(c) for c in self.centers]
        self.v_c = [np.zeros_like(c) for c in self.centers]
        self.m_s = [np.zeros_like(s) for s in self.stds]
        self.v_s = [np.zeros_like(s) for s in self.stds]
        self.t_adam = 0

    # ---------- прямий прохід ----------

    def _membership_all(self, X):
        """
        Обчислює μ_jm(x_j) для всіх зразків.
        Повертає список довжиною N_f; кожен елемент — (N, M) матриця.
        """
        N = X.shape[0]
        mem = []
        for j in range(self.N_f):
            c = self.centers[j][None, :]      # (1, M)
            s = self.stds[j][None, :]         # (1, M)
            xj = X[:, [j]]                     # (N, 1)
            mu = gaussian(xj, c, s)           # (N, M)
            mem.append(mu)
        return mem

    def _rule_firing(self, mem):
        """
        Обчислює ненормовані вогні правил w_k(x) як добуток μ по ознаках.
        mem: список з N_f елементів, кожен (N, M)
        → Wfire: (N, R)
        """
        N = mem[0].shape[0]
        Wfire = np.ones((N, self.R))
        for r, combo in enumerate(self.rule_indices):
            # combo: масив довжини N_f з індексами MF
            prod = 1.0
            for j, m_idx in enumerate(combo):
                prod *= mem[j][:, m_idx]
            Wfire[:, r] = prod
        return Wfire

    def _rule_outputs(self, X):
        # z_k(x) = w_k^T x + b_k, матрично: Z = X @ W^T + b
        return X @ self.W.T + self.b[None, :]   # (N, R)

    def _forward(self, X):
        mem = self._membership_all(X)           # список (N, M)
        Wfire = self._rule_firing(mem)          # (N, R)
        s = np.sum(Wfire, axis=1, keepdims=True) + 1e-12
        beta = Wfire / s                        # нормовані ваги (N, R)
        Z = self._rule_outputs(X)               # (N, R)
        y_hat = np.sum(beta * Z, axis=1)        # (N,)
        return y_hat, mem, Wfire, beta, Z

    # ---------- оновлення наслідків (зважена МНК) ----------

    def _update_consequents_wls(self, X, y, Wfire):
        """
        Мінімізує ∑_n ∑_r beta_{nr} * (z_r(x_n) - y_n)^2
        Еквівалентно багатьом незалежним зваженим МНК для кожного правила r.
        Розв'язуємо параметри [w_r, b_r] з дизайном [X, 1].
        """
        N, R = Wfire.shape
        s = np.sum(Wfire, axis=1, keepdims=True) + 1e-12
        beta = Wfire / s                         # (N, R)

        X1 = np.hstack([X, np.ones((N, 1))])     # (N, N_f+1)
        for r in range(R):
            w_r = beta[:, r]                     # ваги (N,)
            # Зважені X'X і X'y
            WX = X1 * w_r[:, None]              # (N, d)
            A = X1.T @ WX                       # (d, d)
            bvec = X1.T @ (w_r * y)             # (d,)
            # Ридж-стабілізація
            lam = 1e-6
            A += lam * np.eye(A.shape[0])
            theta = np.linalg.solve(A, bvec)    # (d,)
            self.W[r, :] = theta[:-1]
            self.b[r] = theta[-1]

    # ---------- оновлення MF (градієнт + Adam) ----------

    def _update_premise_adam(self, X, y, y_hat, mem, Wfire, Z):
        cfg = self.cfg
        N = X.shape[0]
        s_sum = np.sum(Wfire, axis=1, keepdims=True) + 1e-12
        beta = Wfire / s_sum                    # (N, R)

        # похідна L = 0.5*||y - y_hat||^2
        err = (y_hat - y)                       # (N,)

        # Похідна по Wfire через beta (вплив нормалізації):
        # y_hat = sum_r beta_r * Z_r
        # де beta_r = Wfire_r / sum_k Wfire_k
        # d y_hat / d Wfire_r = (Z_r - y_hat) / sum_k Wfire_k
        dyhat_dW = (Z - y_hat[:, None]) / s_sum  # (N, R)
        dL_dW = err[:, None] * dyhat_dW          # (N, R)

        # Тепер dWfire_r / d μ_{j,m} = прод_інших μ, тобто Wfire_r / μ_{j,m(x)}
        # і d μ / d c, d μ / d s для ґауса
        grad_c = [np.zeros_like(self.centers[j]) for j in range(self.N_f)]
        grad_s = [np.zeros_like(self.stds[j]) for j in range(self.N_f)]

        for r, combo in enumerate(self.rule_indices):
            # внесок правила r
            dL_dW_r = dL_dW[:, r]                              # (N,)
            W_r = Wfire[:, r]                                  # (N,)
            for j, m_idx in enumerate(combo):
                mu_jm = mem[j][:, m_idx]                       # (N,)
                # ∂W_r/∂mu_jm = W_r / mu_jm (де mu_jm>0)
                safe = np.maximum(mu_jm, 1e-12)
                dW_dmu = W_r / safe                            # (N,)
                # ґрадієнти μ по c та s:
                c = self.centers[j][m_idx]
                s = np.maximum(self.stds[j][m_idx], 1e-6)
                xj = X[:, j]
                # μ = exp( -0.5 ((x-c)/s)^2 )
                dmu_dc = mu_jm * ((xj - c) / (s**2))           # (N,)
                dmu_ds = mu_jm * (((xj - c)**2) / (s**3))      # (N,)
                # Ланцюгове правило
                dL_dc = np.sum(dL_dW_r * dW_dmu * dmu_dc)
                dL_ds = np.sum(dL_dW_r * dW_dmu * dmu_ds)
                grad_c[j][m_idx] += dL_dc
                grad_s[j][m_idx] += dL_ds

        # Adam крок
        self.t_adam += 1
        b1, b2, eps = cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps
        lr = cfg.lr

        for j in range(self.N_f):
            # c
            self.m_c[j] = b1 * self.m_c[j] + (1 - b1) * grad_c[j]
            self.v_c[j] = b2 * self.v_c[j] + (1 - b2) * (grad_c[j] ** 2)
            mhat = self.m_c[j] / (1 - b1 ** self.t_adam)
            vhat = self.v_c[j] / (1 - b2 ** self.t_adam)
            self.centers[j] -= lr * mhat / (np.sqrt(vhat) + eps)
            # s (додатково гарантуємо позитивність)
            self.m_s[j] = b1 * self.m_s[j] + (1 - b1) * grad_s[j]
            self.v_s[j] = b2 * self.v_s[j] + (1 - b2) * (grad_s[j] ** 2)
            mhat = self.m_s[j] / (1 - b1 ** self.t_adam)
            vhat = self.v_s[j] / (1 - b2 ** self.t_adam)
            self.stds[j] -= lr * mhat / (np.sqrt(vhat) + eps)
            self.stds[j] = np.maximum(self.stds[j], 1e-4)

    # ---------- API ----------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._init_params(X)

        cfg = self.cfg
        idx = np.arange(X.shape[0])

        for ep in range(cfg.epochs):
            # 1) Оновити наслідки з усього набору (стабільніше)
            y_hat_full, mem_full, Wfire_full, beta_full, Z_full = self._forward(X)
            self._update_consequents_wls(X, y, Wfire_full)

            # 2) Стохастичне оновлення премісів
            np.random.shuffle(idx)
            for start in range(0, len(idx), cfg.batch_size):
                bidx = idx[start:start + cfg.batch_size]
                y_hat, mem, Wfire, beta, Z = self._forward(X[bidx])
                self._update_premise_adam(X[bidx], y[bidx], y_hat, mem, Wfire, Z)

            # Лог
            if cfg.verbose:
                y_hat_full, *_ = self._forward(X)
                loss = 0.5 * np.mean((y - y_hat_full) ** 2)
                print(f"Epoch {ep+1:03d}/{cfg.epochs} - MSE: {loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        y_hat, *_ = self._forward(X)
        return y_hat

    def rules_text(self, feature_names=None):
        """
        Повертає текстове представлення правил:
        IF x1 is Gauss(c,s) AND x2 is Gauss(c,s) ... THEN y = w^T x + b
        """
        if feature_names is None:
            feature_names = [f"x{j+1}" for j in range(self.N_f)]
        lines = []
        for r, combo in enumerate(self.rule_indices):
            parts = []
            for j, m_idx in enumerate(combo):
                c = self.centers[j][m_idx]
                s = self.stds[j][m_idx]
                parts.append(f"{feature_names[j]} is N(c={c:.4f}, s={s:.4f})")
            antecedent = " AND ".join(parts)
            w = self.W[r]
            b = self.b[r]
            lin = " + ".join([f"{w[j]:.4f}*{feature_names[j]}" for j in range(self.N_f)])
            lines.append(f"IF {antecedent} THEN y = {lin} + {b:.4f}")
        return "\n".join(lines)

# --------------------- Демо використання ---------------------

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="ANFIS демо: апроксимація 2D-функції.")
    parser.add_argument("--M", type=int, default=3, help="Кількість MF на ознаку.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Згенеруємо синтетичні дані (можна замінити на власні X,y)
    N_train, N_test = 1500, 400
    def make_xy(N):
        x1 = rng.uniform(-3.0, 3.0, size=N)
        x2 = rng.uniform(-3.0, 3.0, size=N)
        X = np.stack([x1, x2], axis=1)
        y = np.sin(x1) + 0.5*np.cos(x2) + 0.1*rng.normal(size=N)
        return X, y

    Xtr, ytr = make_xy(N_train)
    Xte, yte = make_xy(N_test)

    cfg = ANFISConfig(
        M=args.M, lr=args.lr, epochs=args.epochs, batch_size=args.batch,
        random_state=args.seed, verbose=args.verbose
    )
    model = ANFIS(cfg).fit(Xtr, ytr)

    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)
    mse_tr = np.mean((ytr - yhat_tr)**2)
    mse_te = np.mean((yte - yhat_te)**2)
    print(f"\nTrain MSE: {mse_tr:.6f} | Test MSE: {mse_te:.6f}\n")

    print("=== Приклади правил ===")
    print("\n".join(model.rules_text(feature_names=['x1','x2']).splitlines()[:min(10, model.R)]))

    # Невелика візуалізація уздовж x2=0 (зріз функції)
    xs = np.linspace(-3, 3, 300)
    Xslice = np.stack([xs, np.zeros_like(xs)], axis=1)
    yslice_true = np.sin(xs) + 0.5*np.cos(0.0)
    yslice_pred = model.predict(Xslice)

    plt.figure(figsize=(7,4))
    plt.plot(xs, yslice_true, label="target slice (x2=0)")
    plt.plot(xs, yslice_pred, label="ANFIS prediction")
    plt.xlabel("x1"); plt.ylabel("y")
    plt.title("ANFIS slice prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()
