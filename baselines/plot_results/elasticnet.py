import numpy as np
import matplotlib.pyplot as plt

def prox_elasticnet(b, lambda1, lambda2):
    # The proximal operator of the elastic net
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_):
    # Computes the augmented Lagrangian for elastic net
    term1 = 0.5 * np.linalg.norm(A @ Z - B, 'fro')**2
    term2 = np.linalg.norm(X, 1) + lambda_ * np.linalg.norm(X, 'fro')**2
    term3 = np.sum(Y1 * (A @ Z - B)) + np.sum(Y2 * (X - Z))
    term4 = (mu / 2) * (np.linalg.norm(A @ Z - B, 'fro')**2 + np.linalg.norm(X - Z, 'fro')**2)
    return term1 + term2 + term3 + term4

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho


def elasticnet_accelerated(A, B, lambda_, opts):
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

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_)
    objective_list = []
    errors_list = []
    for iter in range(1, max_iter + 1):
        Xk, Zk = X.copy(), Z.copy()

        # update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_ / mu)
        # update Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        # Calculate objective and error for debug
        obj = np.linalg.norm(X, 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Adaptive mu update based on Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu, lambda_)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2


    return errors_list, objective_list


def elasticnet_admm(A, B, lambda_, opts):
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
    objective_list = []
    errors_list = []
    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        # update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_ / mu)
        # update Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = np.linalg.norm(X.ravel(), 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        errors_list.append(err)
        objective_list.append(obj)
        if chg < tol:
            break


        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)

    return errors_list, objective_list


def elasticnet_fast_admm(A, B, lambda_, opts):
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
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        # update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambda_ / mu)
        # update Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        obj = np.linalg.norm(X.ravel(), 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    return errors_list, objective_list


def elasticnet_autoAdmm(A, B, lambd, opts):
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
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        # update X
        X = prox_elasticnet(Z - Y2 / mu, 1 / mu, lambd / mu)
        # update Z
        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        obj = np.linalg.norm(X.ravel(), 1) + lambda_ * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')

        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Update rho dynamically
        rho_update = 2 * (np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2) / (
                    np.linalg.norm(X - Z, 'fro') ** 2 + np.linalg.norm(A @ Z - B, 'fro') ** 2)
        rho = min(max(rho_update, 1.1), 10)

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)
    return errors_list, objective_list


if __name__ == "__main__":
    # Generate toy data
    d = 10
    na = 200
    nb = 100

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X
    b = B[:, 0]

    # Options for the elastic net minimization
    opts = {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.1,
        'mu': 1e-4,
        'max_mu': 1e10,
        'DEBUG': 1
    }

    # Regularization parameter for elastic net
    lambda_ = 0.01
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = elasticnet_accelerated(A, B, lambda_, opts)
        elif key == "ADMM":
            error_list, objective_list = elasticnet_admm(A, B, lambda_, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = elasticnet_fast_admm(A, B, lambda_, opts)
        else:
            error_list, objective_list = elasticnet_autoAdmm(A, B, lambda_, opts)

        if len(objective_list) < 1000:
            objective_list.extend([objective_list[-1]] * (1000 - len(objective_list)))
        objective_list_map[key] = objective_list

    # Plotting the objective lists with different line styles and markers
    plt.figure(figsize=(10, 6))
    for key, objective_list in objective_list_map.items():
        plt.plot(objective_list, label=key)

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Value Convergence for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig("./elasticnet.pdf")

