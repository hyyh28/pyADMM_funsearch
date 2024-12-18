import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv

def diagAtB(A, B):
    return np.einsum('ij,ij->j', A, B)

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho

def prox_nuclear(B, lambda_):
    U, S, Vt = svd(B, full_matrices=False)
    S_threshold = np.maximum(S - lambda_, 0)
    X = U @ np.diag(S_threshold) @ Vt
    nuclearnorm = np.sum(S_threshold)
    return X, nuclearnorm

def prox_l1(b, lambda_):
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro')**2
    else:
        raise ValueError('Loss function not supported')

def augmented_lagrangian(A, b, x, Z, e, Y1, Y2, mu, lambda_):
    term1 = lambda_ * np.linalg.norm(Z, ord='nuc')
    term2 = np.sum(Y1 * (A @ x + e - b)) + np.sum(Y2 * (A @ np.diag(x) - Z))
    term3 = (mu / 2) * (np.linalg.norm(A @ x + e - b)**2 + np.linalg.norm(A @ np.diag(x) - Z, 'fro')**2)
    return term1 + term2 + term3
    
def tracelassoR_accelerated(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))

    prev_aug_lagrangian = augmented_lagrangian(A, b, x, Z, e, Y1, Y2, mu, lambda_)
    objective_list = []
    errors_list = []

    for iter in range(max_iter):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        # First super block {Z, e}
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / mu, lambda_ / mu)
        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {x}
        x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + diagAtB(A, Y2 / mu + Z))

        # Compute residuals and errors
        dY1 = A @ x + e - b
        dY2 = Z - A @ np.diag(x)
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = comp_loss(e, loss) + lambda_ * nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Adaptive mu update based on augmented Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, b, x, Z, e, Y1, Y2, mu, lambda_)
        lagrangian_diff = current_aug_lagrangian - prev_aug_lagrangian
        prev_aug_lagrangian = current_aug_lagrangian

        if lagrangian_diff > 0:
            mu = min(rho * mu, max_mu)
        else:
            mu /= rho

        # Update dual variables
        Y1 += mu * dY1
        Y2 += mu * dY2
    
    return errors_list, objective_list


def tracelassoR_admm(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))
    objective_list = []
    errors_list = []

    for iter in range(max_iter):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        # first super block {Z,e}
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / mu, lambda_ / mu)
        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        # second super block {x}
        x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + diagAtB(A, Y2 / mu + Z))

        dY1 = A @ x + e - b
        dY2 = Z - A @ np.diag(x)
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = comp_loss(e, loss) + lambda_ * nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)

    
    return errors_list, objective_list

def tracelassoR_fast_admm(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))
    objective_list = []
    errors_list = []

    for iter in range(max_iter):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        # first super block {Z,e}
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / mu, lambda_ / mu)
        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        # second super block {x}
        x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + diagAtB(A, Y2 / mu + Z))

        dY1 = A @ x + e - b
        dY2 = Z - A @ np.diag(x)
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = comp_loss(e, loss) + lambda_ * nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)

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

def tracelassoR_autoAdmm(A, b, lambda_, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')
    rho_update_factor = opts.get('rho_update_factor', 1.1)
    rho_tol = opts.get('rho_tol', 1e-4)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = inv(AtA + np.diag(np.diag(AtA)))
    rho = mu

    objective_list = []
    errors_list = []

    for iter in range(max_iter):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) - Y2 / rho, lambda_ / rho)
        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / rho, 1 / rho)
        elif loss == 'l2':
            e = rho * (b - A @ x - Y1 / rho) / (1 + rho)
        else:
            raise ValueError('Unsupported loss function')

        x = invAtA @ (-A.T @ (Y1 / rho + e) + Atb + diagAtB(A, Y2 / rho + Z))

        dY1 = A @ x + e - b
        dY2 = Z - A @ np.diag(x)
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = comp_loss(e, loss) + lambda_ * nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1)**2 + np.linalg.norm(dY2)**2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, rho={rho}, obj={obj}, err={err}')
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        if np.linalg.norm(dY1) > rho_tol * np.linalg.norm(dY2):
            rho *= rho_update_factor
        elif np.linalg.norm(dY2) > rho_tol * np.linalg.norm(dY1):
            rho /= rho_update_factor

        Y1 = Y1 + rho * dY1
        Y2 = Y2 + rho * dY2
        rho = min(rho, max_mu)
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
    lambda_ = 0.1
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = tracelassoR_accelerated(A, b, lambda_, opts)
        elif key == "ADMM":
            error_list, objective_list = tracelassoR_admm(A, b, lambda_, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = tracelassoR_fast_admm(A, b, lambda_, opts)
        else:
            error_list, objective_list = tracelassoR_autoAdmm(A, b, lambda_, opts)

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
    plt.savefig("./tracelassoR.pdf")

