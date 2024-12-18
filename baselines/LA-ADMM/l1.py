import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, rho, max_mu):
    # Update dual variables
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    # Increase penalty parameter adaptively
    mu = min(mu * beta_factor, max_mu)
    return Y1, Y2, mu, rho

def la_admm_l1(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    beta_factor = opts.get('beta_factor', 2)  # Penalty multiplier per stage
    stage_iter = opts.get('stage_iter', 100)  # Iterations per stage

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

    total_iter = 0
    while total_iter < max_iter:
        for stage in range(max_iter // stage_iter):
            for iter in range(stage_iter):
                Xk = X.copy()
                Zk = Z.copy()
                # update X
                X = prox_l1(Z - Y2 / mu, 1 / mu)
                # update Z
                Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

                dY1 = A @ Z - B
                dY2 = X - Z

                chgX = np.max(np.abs(Xk - X))
                chgZ = np.max(np.abs(Zk - Z))
                chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

                if DEBUG and (total_iter == 1 or total_iter % 10 == 0):
                    obj = np.linalg.norm(X.ravel(), 1)
                    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                    print(f'Stage {stage}, Iter {iter}, mu={mu:.2e}, obj={obj}, err={err}')

                if chg < tol:
                    break

                total_iter += 1
                if total_iter >= max_iter:
                    break

            # After each stage, update penalty parameter
            Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, beta_factor, rho, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, total_iter

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

    # Perform l1 minimization using LA-ADMM
    X2, obj, err, iter = la_admm_l1(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter

datasets = {}

datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'beta_factor': 2,  # Multiplier for penalty parameter
    'stage_iter': 50,  # Number of iterations per stage
    'DEBUG': 1
    },
    'd': 10,
    'na': 200,
    'nb': 100
}

evaluate(datasets['l1'])