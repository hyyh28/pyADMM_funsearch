import numpy as np
import matplotlib.pyplot as plt

def prox_gl1(b, G, lambda_val):
    x = np.zeros_like(b)
    for g in G:
        b_g = b[g]
        norm_b_g = np.linalg.norm(b_g)
        if norm_b_g > lambda_val:
            x[g] = b_g * (1 - lambda_val / norm_b_g)
    return x

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho

def augmented_lagrangian(A, B, X, Z, Y1, Y2, mu):
    primal_residual = np.linalg.norm(A @ Z - B) + np.linalg.norm(X - Z)
    dual_residual = np.linalg.norm(A.T @ Y1 + Y2)
    return primal_residual + dual_residual + (mu / 2) * (primal_residual ** 2 + dual_residual ** 2)

def groupl1_accelerated(A, B, G, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    history_X, history_Z = [], []
    prev_aug_lagrangian = np.inf
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Zk = X.copy(), Z.copy()

        # Update X
        X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T

        # Update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # Compute residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        # Calculate objective and error for debug
        obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Anderson Acceleration
        if len(history_X) == anderson_m:
            history_X.pop(0)
            history_Z.pop(0)
        history_X.append(X.copy())
        history_Z.append(Z.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            G_X, G_Z = delta_X.T, delta_Z.T
            g_X, g_Z = (X - history_X[0]).flatten(), (Z - history_Z[0]).flatten()

            if G_X.shape[0] == g_X.shape[0]:
                alpha_X = np.linalg.lstsq(G_X, g_X, rcond=None)[0]
                X -= np.sum(alpha_X[:, np.newaxis] * delta_X, axis=0).reshape(X.shape)
            if G_Z.shape[0] == g_Z.shape[0]:
                alpha_Z = np.linalg.lstsq(G_Z, g_Z, rcond=None)[0]
                Z -= np.sum(alpha_Z[:, np.newaxis] * delta_Z, axis=0).reshape(Z.shape)

        # Adaptive mu update using augmented Lagrangian
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, Y1, Y2, mu)
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


def groupl1_admm(A, B, G, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = X.copy()
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # update X
        X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T

        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # update residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)

    obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return errors_list, objective_list

def groupl1_fast_admm(A, B, G, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    Z = X.copy()
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros((na, nb))

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        # update X
        X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T

        # update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        # update residuals
        dY1 = A @ Z - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return errors_list, objective_list

def groupl1_autoAdmm(A, B, G, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
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
        
        # Update X
        X = np.array([prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu) for i in range(nb)]).T
        
        # Update Z
        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)
        
        dY1 = A @ Z - B
        dY2 = X - Z
        
        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])
        obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f'iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}')

        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Update rho dynamically
        # rho_update = 2 * (np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2) / (np.linalg.norm(X - Z, 'fro')**2 + np.linalg.norm(A @ Z - B, 'fro')**2)
        # rho = min(max(rho_update, 1.1), 10)

        rho_update_factor = 1.01  # Example factor for updating rho
        rho = min(rho * rho_update_factor, max_mu)
        
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
    # Group L1 Regularization
    g_num = 5
    g_len = round(na / g_num)
    G = [list(range(i * g_len, (i + 1) * g_len)) for i in range(g_num - 1)]
    G.append(list(range((g_num - 1) * g_len, na)))

    # Regularization parameter for elastic net
    lambda_ = 0.01
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = groupl1_accelerated(A, B, G, opts)
        elif key == "ADMM":
            error_list, objective_list = groupl1_admm(A, B, G, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = groupl1_fast_admm(A, B, G, opts)
        else:
            error_list, objective_list = groupl1_autoAdmm(A, B, G, opts)

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
    plt.savefig("./groupl1.pdf")

