import numpy as np
from scipy.linalg import svd

def generate_toy_data(d=10, na=200, nb=100):
    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = np.dot(A, X)
    b = B[:, 0]
    return A, X, B, b

def prox_l1(b, lambd):
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def prox_nuclear(B, lambd):
    U, S, Vh = svd(B, full_matrices=False)
    S = np.maximum(S - lambd, 0)
    X = np.dot(U * S, Vh)
    nuclearnorm = np.sum(S)
    return X, nuclearnorm

def project_simplex(B):
    n, m = B.shape
    B_sort = np.sort(B, axis=1)[:, ::-1]
    cum_B = np.cumsum(B_sort, axis=1)
    A = np.arange(1, m + 1)
    sigma = B_sort - (cum_B - 1) / A
    idx = np.sum(sigma > 0, axis=1)
    sigma = np.take_along_axis(B_sort, np.expand_dims(idx - 1, axis=1), axis=1)
    sigma = np.tile(sigma, (1, m))
    return np.maximum(B - sigma, 0)

def comp_loss(E, loss_type):
    if loss_type == 'l1':
        return np.linalg.norm(E, 1)
    elif loss_type == 'l21':
        return np.sum(np.linalg.norm(E, axis=0))
    elif loss_type == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2

def rmsc(X, lambd, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    debug = opts.get('DEBUG', 0)

    d, n, m = X.shape
    L = np.zeros((d, n))
    S = np.zeros((d, n, m))
    Z = L.copy()
    Y = S.copy()
    Y2 = L.copy()
    
    for iter in range(max_iter):
        Lk, Sk, Zk = L.copy(), S.copy(), Z.copy()
        
        Z, nuclearnormZ = prox_nuclear(L + Y2 / mu, 1 / mu)
        for i in range(m):
            S[:, :, i] = prox_l1(X[:, :, i] - L - Y[:, :, i] / mu, lambd / mu)
        
        L = project_simplex((np.sum(X - S - Y / mu, axis=2) + Z - Y2 / mu) / (m + 1))
        
        dY = np.zeros_like(S)
        for i in range(m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]
        dY2 = L - Z
        
        chgL = np.max(np.abs(Lk - L))
        chgZ = np.max(np.abs(Zk - Z))
        chgS = np.max([np.max(np.abs(Sk[:, :, i] - S[:, :, i])) for i in range(m)])
        chg = max(chgL, chgZ, chgS, np.max(np.abs(dY)), np.max(np.abs(dY2)))
        
        if debug and (iter == 0 or iter % 10 == 0):
            obj = nuclearnormZ + lambd * np.sum(np.abs(S))
            err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        if chg < tol:
            break
        
        Y += mu * dY
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)
    
    obj = nuclearnormZ + lambd * np.sum(np.abs(S))
    err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return L, S, obj, err, iter

# Example usage
d = 10
na = 200
nb = 100
A, X, B, b = generate_toy_data(d, na, nb)

n = 100
r = 5
m = 10
X = np.random.randn(n, n, m)
lambda_val = 1 / np.sqrt(n)
opts = {'tol': 1e-6, 'max_iter': 1000, 'rho': 1.2, 'mu': 1e-3, 'max_mu': 1e10, 'DEBUG': 1}

L, S, obj, err, iter = rmsc(X, lambda_val, opts)

print('Error:', err)
print('Iterations:', iter)
