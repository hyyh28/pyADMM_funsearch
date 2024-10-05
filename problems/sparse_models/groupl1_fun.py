import numpy as np
import matplotlib.pyplot as plt

def prox_gl1(b, G, lambda_val):
    """
    The proximal operator of the group l1 norm

    min_x lambda * sum(||x_g||_2) + 0.5 * ||x - b||_2^2

    Parameters:
        b : numpy.ndarray
            Input vector
        G : list of lists
            A list of groups, where each group is a list of indices
        lambda_val : float
            Regularization parameter

    Returns:
        x : numpy.ndarray
            Output vector after applying the proximal operator
    """
    x = np.zeros_like(b)
    for g in G:
        b_g = b[g]
        norm_b_g = np.linalg.norm(b_g)
        if norm_b_g > lambda_val:
            x[g] = b_g * (1 - lambda_val / norm_b_g)
    return x

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
    Z = np.zeros_like(X)
    Y1 = np.zeros((d, nb))
    Y2 = np.zeros_like(X)

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I) @ I

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
        
        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
            err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
            print(f'iter {iter}, mu={mu}, rho={rho}, obj={obj}, err={err}')
        
        if chg < tol:
            break
        
        # Update dual variables
        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        
        # Update penalty parameter rho dynamically
        rho_update_factor = 1.01  # Example factor for updating rho
        rho = min(rho * rho_update_factor, max_mu)
        
        # Update mu
        mu = min(rho * mu, max_mu)
    
    obj = sum(np.linalg.norm(X[g], axis=1).sum() for g in G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
    
    return X, obj, err, iter

  

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0
}

# 生成toy数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true
b = B[:, 0]

# Group L1正则化
g_num = 5
g_len = round(na / g_num)
G = [list(range(i * g_len, (i + 1) * g_len)) for i in range(g_num - 1)]
G.append(list(range((g_num - 1) * g_len, na)))

X, obj, err, iter = groupl1_admm(A, B, G, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
plt.stem(X[:, 0])
plt.title('Group L1 Regularization Result')
plt.show()


