import numpy as np
import matplotlib.pyplot as plt

def update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu):
    Y1 = Y1 + mu * dY1
    Y2 = Y2 + mu * dY2
    mu = 1
    return Y1, Y2, mu, rho

def prox_elasticnet(b, lambda1, lambda2):
    # The proximal operator of the elastic net
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)

def prox_l1(b, lambda_val):
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss):
    term1 = comp_loss(E, loss)
    term2 = lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
    term3 = np.sum(Y1 * (A @ Z + E - B)) + np.sum(Y2 * (X - Z))
    term4 = (mu / 2) * (np.linalg.norm(A @ Z + E - B, 'fro') ** 2 + np.linalg.norm(X - Z, 'fro') ** 2)
    return term1 + term2 + term3 + term4

def elasticnetR_accelerated(A, B, lambda1, lambda2, opts):
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

    prev_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss)
    objective_list = []
    errors_list = []

    for iter in range(1, max_iter + 1):
        Xk, Ek, Zk = X.copy(), E.copy(), Z.copy()

        # Update X and E
        X = prox_elasticnet(Z - Y2 / mu, lambda1 / mu, lambda2 / mu)
        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Unsupported loss function')

        # Update Z
        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        # Compute residuals and errors
        dY1 = A @ Z + E - B
        dY2 = X - Z
        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = max(chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        # Adaptive mu update based on augmented Lagrangian difference
        current_aug_lagrangian = augmented_lagrangian(A, B, X, Z, E, Y1, Y2, mu, lambda1, lambda2, loss)
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


def elasticnetR_admm(A, B, lambda1, lambda2, opts):
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
        X = prox_elasticnet(Z - Y2 / mu, lambda1 / mu, lambda2 / mu)
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

        obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        Y1, Y2, mu, rho = update_penalty(dY1, dY2, Y1, Y2, mu, rho, max_mu)


    return errors_list, objective_list

def elasticnetR_fast_admm(A, B, lambda1, lambda2, opts):
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
        X = prox_elasticnet(Z - Y2 / mu, lambda1 / mu, lambda2 / mu)
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

        obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
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

    obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return errors_list, objective_list


def elasticnetR_autoAdmm(A, B, lambda1, lambda2, opts):
    tol = opts.get('tol', 1e-6)
    max_iter = opts.get('max_iter', 1000)
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

        X = prox_elasticnet(Z - Y2 / mu, lambda1 / mu, lambda2 / mu)
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
        obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
        err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)

        if DEBUG and (iter == 1 or iter % 10 == 0):
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")
        errors_list.append(err)
        objective_list.append(obj)

        if chg < tol:
            break

        rho_update_factor = 1.01
        rho *= rho_update_factor

        mu = min(rho * mu, max_mu)

        Y1 += mu * dY1
        Y2 += mu * dY2

    obj = comp_loss(E, loss) + lambda1 * np.linalg.norm(X, 1) + lambda2 * np.linalg.norm(X, 'fro') ** 2
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    
    
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
    lambda1 = 10
    lambda2 = 10
    objective_list_map = {"Accelerated": [], "ADMM": [], "Fast ADMM": [], "AutoADMM": []}
    line_styles = {"Accelerated": "-", "ADMM": "--", "Fast ADMM": "-.", "AutoADMM": ":"}  # Different line styles
    markers = {"Accelerated": "o", "ADMM": "s", "Fast ADMM": "^", "AutoADMM": "D"}  # Different markers

    for key in objective_list_map.keys():
        if key == "Accelerated":
            error_list, objective_list = elasticnetR_accelerated(A, B, lambda1, lambda2, opts)
        elif key == "ADMM":
            error_list, objective_list = elasticnetR_admm(A, B, lambda1, lambda2, opts)
        elif key == "Fast ADMM":
            error_list, objective_list = elasticnetR_fast_admm(A, B, lambda1, lambda2, opts)
        else:
            error_list, objective_list = elasticnetR_autoAdmm(A, B, lambda1, lambda2, opts)

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
    plt.savefig("./elasticnetR.pdf")

