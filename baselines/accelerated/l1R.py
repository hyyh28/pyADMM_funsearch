import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambda_val):
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l21':
        return np.sum([np.linalg.norm(E[:, i]) for i in range(E.shape[1])])
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def augmented_lagrangian(A, B, X, Z, E, Y1, Y2, lambda_val, mu, loss):
    return comp_loss(E, loss) + lambda_val * np.linalg.norm(X, 1) + np.sum(Y1 * (A @ Z + E - B)) + np.sum(Y2 * (X - Z)) + (mu / 2) * (np.linalg.norm(A @ Z + E - B, 'fro') ** 2 + np.linalg.norm(X - Z, 'fro') ** 2)

def l1R_adaptive_anderson(A, B, lambda_val, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    history_X, history_Z, history_E = [], [], []

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, lambda_val, mu, loss)

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # First super block {X,E}
        X = prox_l1(Z - Y2 / mu, lambda_val / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(E, loss) + lambda_val * np.linalg.norm(X, 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        # Anderson acceleration
        if len(history_X) == anderson_m:
            history_X.pop(0)
            history_Z.pop(0)
            history_E.pop(0)
        
        history_X.append(X.copy())
        history_Z.append(Z.copy())
        history_E.append(E.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            delta_E = np.array([history_E[i] - history_E[i - 1] for i in range(1, len(history_E))]).reshape(len(history_E) - 1, -1)

            G_X = delta_X.T
            G_Z = delta_Z.T
            G_E = delta_E.T
            g_X = (X - history_X[0]).flatten()
            g_Z = (Z - history_Z[0]).flatten()
            g_E = (E - history_E[0]).flatten()

            if G_X.shape[0] == g_X.shape[0]:  # Ensure compatibility before lstsq
                alpha_X = np.linalg.lstsq(G_X, g_X, rcond=None)[0]
                X -= np.sum(alpha_X[:, np.newaxis] * delta_X, axis=0).reshape(X.shape)
            if G_Z.shape[0] == g_Z.shape[0]:
                alpha_Z = np.linalg.lstsq(G_Z, g_Z, rcond=None)[0]
                Z -= np.sum(alpha_Z[:, np.newaxis] * delta_Z, axis=0).reshape(Z.shape)
            if G_E.shape[0] == g_E.shape[0]:
                alpha_E = np.linalg.lstsq(G_E, g_E, rcond=None)[0]
                E -= np.sum(alpha_E[:, np.newaxis] * delta_E, axis=0).reshape(E.shape)

        # Adaptive mu update
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, lambda_val, mu, loss)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2

    obj = comp_loss(E, loss) + lambda_val * np.linalg.norm(X, 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 1,
    'loss': 'l1',
    'anderson_m': 5
}

# 生成玩具数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# regularized l1
lambda_val = 0.01
X, E, obj, err, iter = l1R_adaptive_anderson(A, B, lambda_val, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('Regularized L1 Norm Result')
# plt.show()