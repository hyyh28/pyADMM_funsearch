import numpy as np
import matplotlib.pyplot as plt

def prox_gl1(b, G, lambda_val):
    x = np.zeros_like(b)
    for g in G:
        nxg = np.linalg.norm(b[g])
        if nxg > lambda_val:
            x[g] = b[g] * (1 - lambda_val / nxg)
    return x

def prox_l1(b, lambda_val):
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l21':
        return np.sum([np.linalg.norm(E[:, i]) for i in range(E.shape[1])])
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def compute_groupl1(X, G):
    obj = 0
    for i in range(X.shape[1]):
        x = X[:, i]
        for g in G:
            obj += np.linalg.norm(x[g])
    return obj

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho

def augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda_val, G, loss):
    residual_1 = A @ Z + E - B
    residual_2 = X - Z
    penalty_term = lambda_val * compute_groupl1(X, G)
    lagrangian = comp_loss(E, loss) + penalty_term \
                 + np.sum(Y1 * residual_1) + np.sum(Y2 * residual_2) \
                 + 0.5 * mu * (np.linalg.norm(residual_1, 'fro') ** 2 + np.linalg.norm(residual_2, 'fro') ** 2)
    return lagrangian

def groupl1R_accelerated(A, B, G, lambda_val, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')
    anderson_m = opts.get('anderson_m', 5)

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    history_X, history_Z, history_E = [], [], []
    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda_val, G, loss)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # First super block {X, E}
        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
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
            history_E.pop(0)
        history_X.append(X.copy())
        history_Z.append(Z.copy())
        history_E.append(E.copy())

        if len(history_X) > 1:
            delta_X = np.array([history_X[i] - history_X[i - 1] for i in range(1, len(history_X))]).reshape(len(history_X) - 1, -1)
            delta_Z = np.array([history_Z[i] - history_Z[i - 1] for i in range(1, len(history_Z))]).reshape(len(history_Z) - 1, -1)
            delta_E = np.array([history_E[i] - history_E[i - 1] for i in range(1, len(history_E))]).reshape(len(history_E) - 1, -1)

            G_X, G_Z, G_E = delta_X.T, delta_Z.T, delta_E.T
            g_X, g_Z, g_E = (X - history_X[0]).flatten(), (Z - history_Z[0]).flatten(), (E - history_E[0]).flatten()

            if G_X.shape[0] == g_X.shape[0]:
                alpha_X = np.linalg.lstsq(G_X, g_X, rcond=None)[0]
                X -= np.sum(alpha_X[:, np.newaxis] * delta_X, axis=0).reshape(X.shape)
            if G_Z.shape[0] == g_Z.shape[0]:
                alpha_Z = np.linalg.lstsq(G_Z, g_Z, rcond=None)[0]
                Z -= np.sum(alpha_Z[:, np.newaxis] * delta_Z, axis=0).reshape(Z.shape)
            if G_E.shape[0] == g_E.shape[0]:
                alpha_E = np.linalg.lstsq(G_E, g_E, rcond=None)[0]
                E -= np.sum(alpha_E[:, np.newaxis] * delta_E, axis=0).reshape(E.shape)

        # Adaptive penalty update based on augmented Lagrangian increment
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda_val, G, loss)
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

def groupl1R_admm(A, B, G, lambda_val, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()
        # First super block {X,E}
        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)
    return errors_list, objective_list

def groupl1R_fast_admm(A, B, G, lambda_val, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()
        # First super block {X,E}
        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Second super block {Z}
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
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

    obj = comp_loss(E, loss) + lambda_val * compute_groupl1(X, G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return errors_list, objective_list

def groupl1R_autoAdmm(A, B, G, lambd, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    min_mu = opts.get('min_mu', 1e-10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, na = A.shape
    _, nb = B.shape

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = np.zeros_like(X)
    Y1 = np.zeros_like(E)
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))
        obj = comp_loss(E, loss) + lambd * compute_groupl1(X, G)
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Update penalty parameter rho dynamically
        rho_update_factor = 1.01  # Example factor for updating rho
        rho = min(rho * rho_update_factor, max_mu)

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = max(min(rho * mu, max_mu), min_mu)
    
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
    lambda_val = 1
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = groupl1R_accelerated(A, B, G, lambda_val, opts)
        elif key == "ADMM":
            error_list, objective_list = groupl1R_admm(A, B, G, lambda_val, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = groupl1R_fast_admm(A, B, G, lambda_val, opts)
        else:
            error_list, objective_list = groupl1R_autoAdmm(A, B, G, lambda_val, opts)

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
    plt.savefig("./groupl1R.pdf")

