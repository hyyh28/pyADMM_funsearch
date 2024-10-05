import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def lagrangian(X, Z, Y1, Y2, A, B, mu):
    # Compute the augmented Lagrangian
    primal_residual = A @ Z - B
    dual_residual = X - Z
    return 0.5 * np.linalg.norm(primal_residual, 'fro')**2 + 0.5 * np.linalg.norm(dual_residual, 'fro')**2 + \
           np.sum(Y1 * primal_residual) + np.sum(Y2 * dual_residual) + \
           (mu / 2) * (np.linalg.norm(primal_residual, 'fro')**2 + np.linalg.norm(dual_residual, 'fro')**2)

def l1(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho_inc = opts.get('rho_inc', 1.05)  # increase factor for mu
    rho_dec = opts.get('rho_dec', 1.02)  # decrease factor for mu
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    min_mu = opts.get('min_mu', 1e-10)  # prevent mu from becoming too small
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        # update X
        X = prox_l1(Z - Y2 / mu, 1 / mu)
        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # Compute Lagrangian and its change
        Lk = lagrangian(Xk, Zk, Y1, Y2, A, B, mu)  # Lagrangian at previous step
        Lk1 = lagrangian(X, Z, Y1, Y2, A, B, mu)  # Lagrangian at current step
        delta_L = Lk1 - Lk  # Increment of Lagrangian
        
        # Compute the partial derivative of L w.r.t mu
        primal_residual = A @ Z - B
        dual_residual = X - Z
        dL_dmu = np.linalg.norm(primal_residual, 'fro')**2 + np.linalg.norm(dual_residual, 'fro')**2

        # Self-adaptive update of mu based on delta_L and its derivative
        if delta_L > 0 and dL_dmu > 0:
            mu = max(mu / rho_dec, min_mu)  # decrease mu
        elif delta_L < 0 and dL_dmu < 0:
            mu = min(rho_inc * mu, max_mu)  # increase mu

        # Update Lagrange multipliers
        Y1 = Y1 + mu * primal_residual
        Y2 = Y2 + mu * dual_residual

        # Check convergence
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(primal_residual)), np.max(np.abs(dual_residual))])

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(primal_residual, 'fro') ** 2 + np.linalg.norm(dual_residual, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, delta_L={delta_L}, dL_dmu={dL_dmu}, err={err}')

        if chg < tol:
            break

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(primal_residual, 'fro') ** 2 + np.linalg.norm(dual_residual, 'fro') ** 2)

    return X, obj, err, iter

def evaluate(instances: dict) -> float:
    # Generate toy data
    d = instances['d']
    na = instances['na']
    nb = instances['nb']

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X

    # Options for the l1 minimization
    opts = instances['opts']

    # Perform l1 minimization
    X2, obj, err, iter = l1(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter


datasets = {}

datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho_inc': 1.05,
    'rho_dec': 1.02,
    'mu': 1e-4,
    'max_mu': 1e10,
    'min_mu': 1e-10,
    'DEBUG': 1
},
'd': 100,
'na': 200,
'nb': 100}

evaluate(datasets['l1'])