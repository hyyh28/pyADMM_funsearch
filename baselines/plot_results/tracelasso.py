import numpy as np
import matplotlib.pyplot as plt

def prox_nuclear(B, lambda_val):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S_threshold = np.maximum(S - lambda_val, 0)
    X = np.dot(U, np.dot(np.diag(S_threshold), Vt))
    nuclearnorm = np.sum(S_threshold)
    return X, nuclearnorm

def diagAtB(A, B):
    return np.einsum('ij,ij->j', A, B)

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho

def augmented_lagrangian(A, b, x, Z, Y1, Y2, mu):
    term1 = np.linalg.norm(Z, ord='nuc')
    term2 = np.sum(Y1 * (A @ x - b)) + np.sum(Y2 * (A @ np.diag(x) - Z))
    term3 = (mu / 2) * (np.linalg.norm(A @ x - b) ** 2 + np.linalg.norm(A @ np.diag(x) - Z, 'fro') ** 2)
    return term1 + term2 + term3

def tracelasso_accelerated(A, b, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))

    prev_aug_lagrangian = augmented_lagrangian(A, b, x, Z, Y1, Y2, mu)

    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()

        # Update x
        x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))

        # Update Z
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

        # Compute residuals and check for convergence
        dY1 = A @ x - b
        dY2 = A @ np.diag(x) - Z
        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        obj = nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Adaptive mu update based on augmented Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, b, x, Z, Y1, Y2, mu)
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


def tracelasso_admm(A, b, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()
        # Update x
        x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))
        # Update Z
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

        # Compute residuals
        dY1 = A @ x - b
        dY2 = A @ np.diag(x) - Z
        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)

    return errors_list, objective_list

def tracelasso_fast_admm(A, b, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()
        # Update x
        x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))
        # Update Z
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

        # Compute residuals
        dY1 = A @ x - b
        dY2 = A @ np.diag(x) - Z
        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    return errors_list, objective_list

def tracelasso_autoAdmm(A, b, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    tau_incr = opts.get('tau_incr', 2)
    tau_decr = opts.get('tau_decr', 2)

    d, n = A.shape
    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = np.zeros((d, n))
    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()

        x = invAtA @ (-A.T @ Y1 / mu + Atb + diagAtB(A, -Y2 / mu + Z))
        Z, nuclearnorm = prox_nuclear(A @ np.diag(x) + Y2 / mu, 1 / mu)

        dY1 = A @ x - b
        dY2 = A @ np.diag(x) - Z
        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        r_norm = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
        s_norm = mu * np.sqrt(np.linalg.norm(x - xk) ** 2 + np.linalg.norm(Z - Zk) ** 2)

        if r_norm > tau_incr * s_norm:
            rho *= tau_incr
            Y1 /= tau_incr
            Y2 /= tau_incr
        elif s_norm > tau_decr * r_norm:
            rho /= tau_decr
            Y1 *= tau_decr
            Y2 *= tau_decr

        obj = nuclearnorm
        err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
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
            error_list, objective_list = tracelasso_accelerated(A, b, opts)
        elif key == "ADMM":
            error_list, objective_list = tracelasso_admm(A, b, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = tracelasso_fast_admm(A, b, opts)
        else:
            error_list, objective_list = tracelasso_autoAdmm(A, b, opts)

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
    plt.savefig("./tracelasso.pdf")

