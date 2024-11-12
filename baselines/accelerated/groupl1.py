import numpy as np
import matplotlib.pyplot as plt

def prox_gl1(b, G, lambda_val):
    x = np.zeros_like(b)
    for g in G:
        b_g = b[g]
        norm_b_g = np.linalg.norm(b_g)
        if norm_b_g > lambda_val:
            x[g] = b_g * (1 - lambda_val / norm_b_g)
    return x

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu):
    primal_residual = np.linalg.norm(A @ Z - B) + np.linalg.norm(X - Z)
    dual_residual = np.linalg.norm(A.T @ Y1 + Y2)
    return primal_residual + dual_residual + (mu / 2) * (primal_residual ** 2 + dual_residual ** 2)

def group_l1_adaptive_anderson(A, B, G, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    history_X, history_Z = [], []
    prev_aug_lagrangian = np.inf

    for iter in range(1, max_iter + 1):
        Xk, Zk = X.copy(), Z.copy()

        # Update X
        X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T

        # Update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        # Calculate objective and error for debug
        obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        # Anderson Acceleration
        if len(history_X) == anderson_m:
            history_X.pop(0)
            history_Z.pop(0)
        history_X.append(X.copy())
        history_Z.append(Z.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            G_X, G_Z = delta_X.T, delta_Z.T
            g_X, g_Z = (X - history_X[0]).flatten(), (Z - history_Z[0]).flatten()

            if G_X.shape[0] == g_X.shape[0]:
                alpha_X = np.linalg.lstsq(G_X, g_X, rcond=None)[0]
                X -= np.sum(alpha_X[:, np.newaxis] * delta_X, axis=0).reshape(X.shape)
            if G_Z.shape[0] == g_Z.shape[0]:
                alpha_Z = np.linalg.lstsq(G_Z, g_Z, rcond=None)[0]
                Z -= np.sum(alpha_Z[:, np.newaxis] * delta_Z, axis=0).reshape(Z.shape)

        # Adaptive mu update using augmented Lagrangian
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2

    return X, obj, err, iter

# Parameters
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 1,
    'anderson_m': 5
}

# Generate data
d, na, nb = 10, 200, 100
A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# Group L1 Regularization
g_num = 5
g_len = round(na / g_num)
G = [list(range(i * g_len, (i + 1) * g_len)) for i in range(g_num - 1)]
G.append(list(range((g_num - 1) * g_len, na)))

X, obj, err, iter = group_l1_adaptive_anderson(A, B, G, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")