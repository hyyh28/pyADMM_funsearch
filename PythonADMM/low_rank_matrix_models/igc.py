import numpy as np

# Proximal operator for nuclear norm
def prox_nuclear(B, lambda_):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S = np.maximum(S - lambda_, 0)
    svp = np.sum(S > 0)
    if svp >= 1:
        S = S[:svp]
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

# Proximal operator for L1 norm
def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

# Project a point onto a box
def project_box(b, l, u):
    return np.maximum(l, np.minimum(b, u))

# Improved Graph Clustering (IGC) function
def igc(A, C, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    C = np.abs(C)
    d, n = A.shape

    L = np.zeros((d, n))
    S = np.zeros_like(L)
    Z = np.zeros_like(L)
    Y1 = np.zeros_like(L)
    Y2 = np.zeros_like(L)

    for iter in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()

        # First super block {L, S}
        L, nuclearnormL = prox_nuclear(Z - Y2 / mu, 1 / mu)
        S = prox_l1(-Z + A - Y1 / mu, C * (lambda_ / mu))

        # Second super block {Z}
        Z = project_box((-S + A + L + (Y2 - Y1) / mu) / 2, 0, 1)

        dY1 = Z + S - A
        dY2 = L - Z
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgL, chgS, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormL + lambda_ * np.sum(C * np.abs(S))
                err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormL + lambda_ * np.sum(C * np.abs(S))
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)

    return L, S, obj, err, iter

# 示例代码，生成测试数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.2,
    'mu': 1e-3,
    'max_mu': 1e10,
    'DEBUG': 0
}

# Improved Graph Clustering (IGC)
n = 100
r = 5
X = np.random.randn(n, r) @ np.random.randn(r, n)
C = np.random.rand(*X.shape)
lambda_ = 1 / np.sqrt(n)
opts['loss'] = 'l1'
opts['DEBUG'] = 1

L, S, obj, err, iter = igc(X, C, lambda_, opts)

print(f'Error: {err}')
print(f'Iterations: {iter}')
