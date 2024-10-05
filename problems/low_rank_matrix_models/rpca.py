import numpy as np

def rpca(X, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = X.shape
    L = np.zeros((d, n))
    S = np.zeros((d, n))
    Y = np.zeros((d, n))

    for iter in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()

        # update L
        L, nuclearnormL = prox_nuclear(-S + X - Y / mu, 1 / mu)

        # update S
        if loss == 'l1':
            S = prox_l1(-L + X - Y / mu, lambda_ / mu)
        elif loss == 'l21':
            S = prox_l21(-L + X - Y / mu, lambda_ / mu)
        else:
            raise ValueError('Unsupported loss function')

        dY = L + S - X
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chg = np.max([chgL, chgS, np.max(np.abs(dY))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormL + lambda_ * comp_loss(S, loss)
                err = np.linalg.norm(dY, 'fro')
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y += mu * dY
        mu = min(rho * mu, max_mu)

    obj = nuclearnormL + lambda_ * comp_loss(S, loss)
    err = np.linalg.norm(dY, 'fro')

    return L, S, obj, err, iter

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

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def prox_l21(B, lambda_):
    X = np.zeros_like(B)
    for i in range(B.shape[1]):
        nxi = np.linalg.norm(B[:, i])
        if nxi > lambda_:
            X[:, i] = (1 - lambda_ / nxi) * B[:, i]

    return X

def comp_loss(E, loss):
    if loss == 'l1':
        return np.sum(np.abs(E))
    elif loss == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    else:
        raise ValueError('Unsupported loss function')

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

n1, n2 = 100, 200
r = 10
L = np.random.randn(n1, r) @ np.random.randn(r, n2)

p = 0.1
m = int(p * n1 * n2)
I = np.random.choice(n1 * n2, m, replace=False)
Omega = np.zeros((n1, n2))
Omega.flat[I] = 1
E = np.sign(np.random.randn(n1, n2) - 0.5)
S = Omega * E
Xn = L + S

lambda_ = 1 / np.sqrt(max(n1, n2))
opts['loss'] = 'l1'
opts['DEBUG'] = 1

Lhat, Shat, obj, err, iter = rpca(Xn, lambda_, opts)

rel_err_L = np.linalg.norm(L - Lhat, 'fro') / np.linalg.norm(L, 'fro')
rel_err_S = np.linalg.norm(S - Shat, 'fro') / np.linalg.norm(S, 'fro')

print(f'rel_err_L: {rel_err_L}')
print(f'rel_err_S: {rel_err_S}')
print(f'Final err: {err}')
print(f'Total iterations: {iter}')
print(f'Objective: {obj}')
print(f'Error: {err}')
print(f'Iterations: {iter}')
