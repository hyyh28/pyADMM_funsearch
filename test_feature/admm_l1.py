import numpy as np

def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def l1_standard(A, B, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.0)  # ADMM penalty parameter
    DEBUG = opts.get('DEBUG', 1)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y = np.zeros((d, nb))

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # update X with proximal operator
        X = prox_l1(Z - (A.T @ Y / rho), 1 / rho)

        # update Z with a least squares solution
        Z = np.linalg.solve(A.T @ A + rho * np.eye(na), A.T @ (B - Y) + rho * X)

        # update the dual variable Y
        Y = Y + rho * (A @ Z - B)

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chgY = np.max(np.abs(A @ Z - B))

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(chgY ** 2 + chgZ ** 2 + chgX ** 2)
            print(f'iter {iter}, rho={rho}, obj={obj}, err={err}')

        if chgX < tol and chgZ < tol and chgY < tol:
            break
        rho_update = 1.0 + 0.1 * (chgX / chgZ)
        rho = min(max(rho * rho_update, 1.0), 2)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(chgY ** 2 + chgZ ** 2 + chgX ** 2)

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
    opts = instances

    # Perform l1 minimization
    X2, obj, err, iter = l1_standard(A, B, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter

datasets = {}


datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
},
'd': 10,
'na': 200,
'nb': 100}


evaluate(datasets['l1'])