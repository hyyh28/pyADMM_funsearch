import numpy as np
import matplotlib.pyplot as plt

def prox_l1(b, lambd):
    return np.sign(b) * np.maximum(np.abs(b) - lambd, 0)

def l1(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
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
        
        dY1 = A @ Z - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        data_fitting_term = 0.5 * np.linalg.norm(A @ X - B, 'fro') ** 2
        regularization_term = np.linalg.norm(X.ravel(), 1)
        objective_function = data_fitting_term + regularization_term
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={objective_function}, err={err}')
            print(f'\tData Fitting Term: {data_fitting_term}, Regularization Term: {regularization_term}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter


def evaluate(instances: dict) -> float:
    # Generate toy data
    d = instances['d']
    na = instances['na']
    nb = instances['nb']

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X
    b = B[:, 0]

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
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
},
'd': 100,
'na': 200,
'nb': 100}


evaluate(datasets['l1'])