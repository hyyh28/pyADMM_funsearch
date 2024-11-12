import numpy as np
import matplotlib.pyplot as plt


def prox_l1(b, lambd):
    # The proximal operator of the l1 norm
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)


def l1_fast(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    log_error = []

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

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}')
            log_error.append(err)

        if chg < tol:
            break

        # Update rho dynamically
        rho_update = 1.01 + 0.1 * (chgX / chgZ)
        rho = min(max(rho * rho_update, 1.1), 10.0)

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter, log_error


def l1_madmm(A, B, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    log_error = []


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

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
            log_error.append(err)

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter, log_error

def l1_admm(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.0)  # ADMM penalty parameter
    DEBUG = opts.get('DEBUG', 0)
    log_error = []

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
            log_error.append(err)

        if chgX < tol and chgZ < tol and chgY < tol:
            break

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(chgY ** 2 + chgZ ** 2 + chgX ** 2)

    return X, obj, err, iter, log_error


def l1_funsearch(A, B, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    log_error = []


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

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = np.linalg.norm(X.ravel(), 1)
            err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
            log_error.append(err)

        if chg < tol:
            break

        # Adjust rho dynamically
        rho_update = 1.0 + 0.1 * (np.linalg.norm(dY1, 'fro') / np.linalg.norm(dY2, 'fro'))
        rho = min(max(rho * rho_update, 1.1), 10.0)

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = np.linalg.norm(X.ravel(), 1)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

    return X, obj, err, iter, log_error


def evaluate(instances: dict) -> None:
    # Generate toy data
    d = instances['d']
    na = instances['na']
    nb = instances['nb']

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X

    # Options for the l1 minimization
    opts = instances['opts']

    # Store results for plotting
    results = {
        'l1_fast': [],
        'l1_madmm': [],
        'l1_admm': [],
        'l1_funsearch': []
    }

    # Define a helper function to collect iteration and error data
    def collect_data(algorithm, name):
        _, obj, err, iter_count, log_errors = algorithm(A, B, opts)
        print(f'{name}: Iterations: {iter_count}, Objective: {obj}, Error: {err}')
        return iter_count, log_errors

    # Collect data for each algorithm
    algorithms = [l1_fast, l1_madmm, l1_admm, l1_funsearch]
    for alg, name in zip(algorithms, results.keys()):
        iter_count, log_errors = collect_data(alg, name)
        results[name] = (iter_count, log_errors)

    # Plotting
    plt.figure(figsize=(10, 6))
    for name, (iters, errs) in results.items():
        # 使用实际记录的log_errors长度作为x轴，并将其乘以10
        x_values = range(1, len(errs) + 1)
        x_values_scaled = [x * 10 for x in x_values]

        # 只绘制前200个step
        x_values_to_plot = x_values_scaled[:20]
        errs_to_plot = errs[:20]

        plt.plot(x_values_to_plot, errs_to_plot, label=name)

    plt.xlabel('Scaled Iterations')
    plt.ylabel('Error')
    plt.title('Training Curves for L1 Minimization Algorithms (First 200 Steps)')
    plt.legend()
    plt.show()


datasets = {}

datasets['l1'] = {'opts': {
    'tol': 1e-6,
    'max_iter': 1000,
    'rho': 1.1,
    'mu': 1e-4,
    'max_mu': 1e10,
    'DEBUG': 1
},
    'd': 10,
    'na': 200,
    'nb': 100}

evaluate(datasets['l1'])