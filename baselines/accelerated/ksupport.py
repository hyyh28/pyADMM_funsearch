import numpy as np
import matplotlib.pyplot as plt

def prox_ksupport(v, k, lambda_val):
    L = 1 / lambda_val
    d = len(v)
    if k >= d:
        return L * v / (1 + L)
    elif k <= 1:
        k = 1

    z = np.sort(np.abs(v))[::-1]
    z *= L
    ar = np.cumsum(z)
    z = np.append(z, -np.inf)
    diff = 0
    err = np.inf
    found = False

    for r in range(k - 1, -1, -1):
        l, T = bsearch(z, ar, k - r, d, diff, k, r, L)
        if ((L + 1) * T >= (l - k + (L + 1) * r + L + 1) * z[k - r]) and \
           (((k - r - 1 == 0) or (L + 1) * T < (l - k + (L + 1) * r + L + 1) * z[k - r - 1])):
            found = True
            break
        diff += z[k - r]
        err_tmp = max(0, (l - k + (L + 1) * r + L + 1) * z[k - r] - (L + 1) * T) + \
                  max(0, - (l - k + (L + 1) * r + L + 1) * z[k - r - 1] + (L + 1) * T)
        if err > err_tmp:
            err_r, err_l, err_T, err = r, l, T, err_tmp

    if not found:
        r, l, T = err_r, err_l, err_T

    p = np.zeros(d)
    if k - r - 1 > 0:
        p[:k - r - 1] = z[:k - r - 1] / (L + 1)
    p[k - r - 1:l + 1] = T / (l - k + (L + 1) * r + L + 1)

    if l + 1 < d:
        p[l + 1:] = z[l + 1:d]  # Ensure the right size for p

    ind = np.argsort(np.abs(v))[::-1]
    rev = np.zeros_like(ind)
    rev[ind] = np.arange(d)

    p = np.sign(v) * p[rev]
    return v - 1 / L * p

def bsearch(z, array, low, high, diff, k, r, L):
    if z[low] == 0:
        return low, 0
    while low < high:
        mid = (low + high) // 2 + 1
        tmp = mid - k + r + 1 + L * (r + 1)
        if z[mid] * tmp - (array[mid] - diff) > 0:
            low = mid
        else:
            high = mid - 1
    return low, array[low] - diff

def ksupport_adaptive_anderson(A, B, k, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
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

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # Update X with prox_ksupport
        temp = Z - Y2 / mu
        temp = prox_ksupport(temp.flatten(), k, 1 / mu)
        X = temp.reshape(na, nb)

        # Update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
            print(f"iter {iter}, mu={mu}, err={err}")

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

        # Adaptive penalty update based on residuals
        r_norm = np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2
        if r_norm > tol:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        Y1 += mu * dY1
        Y2 += mu * dY2

    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return X, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 1,
    'anderson_m': 5
}

# 生成toy数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true

# k-support norm 正则化
k = 10
X, err, iter = ksupport_adaptive_anderson(A, B, k, opts)
print(f"Final Iteration: {iter}, Error: {err}")
# plt.stem(X[:, 0])
# plt.title('k-support Norm Regularization Result')
# plt.show()