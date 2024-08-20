import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

def fused_l1(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n = A.shape
    x = np.zeros(n)
    z = np.zeros_like(x)
    Y1 = np.zeros(d)
    Y2 = np.zeros_like(x)

    Atb = A.T @ b
    I = np.eye(n)
    invAtAI = inv(A.T @ A + I) @ I

    iter = 0
    for iter in range(max_iter):
        xk = x.copy()
        zk = z.copy()
        # update x
        x = flsa(z - Y2 / mu, 1 / mu, lambda_ / mu, n)
        # update z
        z = invAtAI @ (-A.T @ Y1 / mu + Atb + Y2 / mu + x)
        dY1 = A @ z - b
        dY2 = x - z
        chgx = np.max(np.abs(xk - x))
        chgz = np.max(np.abs(zk - z))
        chg = max([chgx, chgz, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_fused_l1(x, 1, lambda_)
            err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        if chg < tol:
            break
        
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)
    
    obj = comp_fused_l1(x, 1, lambda_)
    err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)
    
    return x, obj, err, iter

def flsa(v, lambda1, lambda2, n):
    # Solve the fused Lasso signal approximator (FLSA)
    x = np.copy(v)
    for _ in range(50):  # max_step = 50
        x_old = np.copy(x)
        z = np.clip(x[:-1] - x[1:], -lambda2, lambda2)
        x[:-1] = v[:-1] - z
        x[1:] = v[1:] + z
        x = np.sign(x) * np.maximum(np.abs(x) - lambda1, 0)
        if np.linalg.norm(x - x_old, ord=np.inf) < 1e-10:  # tol2 = 1e-10
            break
    return x

def comp_fused_l1(x, lambda1, lambda2):
    f = lambda1 * np.sum(np.abs(x))
    f += lambda2 * np.sum(np.abs(np.diff(x)))
    return f

# Generate toy data
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X = np.random.randn(na, nb)
B = A @ X
b = B[:, 0]

# Options for the fused L1 minimization
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 0
}

# Regularization parameter for fused L1
lambda_ = 0.01

# Perform fused L1 minimization
x, obj, err, iter = fused_l1(A, b, lambda_, opts)

# Plot the solution vector x
plt.stem(x)
plt.title('fusedL1 Result')
plt.show()
