import numpy as np
import matplotlib.pyplot as plt

def flsa(x, z, v, z0, lambda1, lambda2, n, maxStep, tol, tau, flag):
    """
    The algorithm for solving the Fused Lasso Signal Approximator (FLSA) problem

    Parameters:
        x : numpy.ndarray
            The solution to problem (1)
        z : numpy.ndarray
            The solution to problem (4)
        v : numpy.ndarray
            The input vector to be projected
        z0 : numpy.ndarray
            A guess of the solution of z
        lambda1 : float
            The regularization parameter
        lambda2 : float
            The regularization parameter
        n : int
            The length of v and x
        maxStep : int
            The maximal allowed iteration steps
        tol : float
            The tolerance parameter
        tau : int
            The program sfa is checked every tau iterations for termination
        flag : int
            The flag for initialization and deciding calling sfa

    Returns:
        x : numpy.ndarray
            Updated x after applying flsa
    """
    nn = n - 1
    Av = np.zeros(nn)

    # Compute Av
    Av = v[1:] - v[:-1]

    # Solve the linear system via Thomas's algorithm (or Rose's algorithm)
    zMax = np.linalg.norm(Av)  # Placeholder for Thomas's algorithm implementation

    # First case: lambda2 >= zMax
    if lambda2 >= zMax:
        temp = np.mean(v)
        temp = np.sign(temp) * max(0, abs(temp) - lambda1)
        x = np.full_like(x, temp)
        return x

    # Second case: lambda2 < zMax
    # Initialize z and other variables
    m = flag // 10
    if m == 0:
        z = np.clip(z0, -lambda2, lambda2)
    else:
        if lambda2 >= 0.5 * zMax:
            z = np.clip(z, -lambda2, lambda2)
        else:
            z = np.zeros(nn)

    if flag >= 1 and flag <= 4:
        zz = np.zeros(nn)

    # Placeholder for sfa, sfa_special, sfa_one function calls
    gap = 0
    iterStep = 0
    numS = 0

    # Soft thresholding by lambda1
    x = np.sign(x) * np.maximum(0, np.abs(x) - lambda1)
    return x

def prox_l1(b, lambda_val):
    """
    The proximal operator of the l1 norm

    min_x lambda * ||x||_1 + 0.5 * ||x - b||_2^2

    Parameters:
        b : numpy.ndarray
            Input vector or matrix
        lambda_val : float
            Regularization parameter

    Returns:
        x : numpy.ndarray
            Output vector or matrix after applying the proximal operator
    """
    return np.maximum(0, b - lambda_val) + np.minimum(0, b + lambda_val)

def comp_loss(E, loss):
    """
    Compute the loss for a given matrix E and loss type

    Parameters:
        E : numpy.ndarray
            Error matrix
        loss : str
            Type of loss ('l1', 'l21', 'l2')

    Returns:
        out : float
            Computed loss value
    """
    if loss == 'l1':
        return np.linalg.norm(E, 1)
    elif loss == 'l21':
        return np.sum([np.linalg.norm(E[:, i]) for i in range(E.shape[1])])
    elif loss == 'l2':
        return 0.5 * np.linalg.norm(E, 'fro') ** 2
    else:
        raise ValueError('Unsupported loss function')

def comp_fusedl1(x, lambda1, lambda2):
    """
    Compute the fused l1 norm of vector x

    Parameters:
        x : numpy.ndarray
            Input vector
        lambda1 : float
            Regularization parameter for l1 norm
        lambda2 : float
            Regularization parameter for fused l1 norm

    Returns:
        f : float
            Fused l1 norm value
    """
    f = lambda1 * np.linalg.norm(x, 1)
    f += lambda2 * np.sum(np.abs(np.diff(x)))
    return f

def fusedl1R(A, b, lambda1, lambda2, opts):
    """
    Solve the fused Lasso regularized minimization problem by ADMM

    min_{x,e} loss(e) + lambda1 * ||x||_1 + lambda2 * sum_{i=2}^p |x_i - x_{i-1}|,
    loss(e) = ||e||_1 or 0.5 * ||e||_2^2

    Parameters:
        A : numpy.ndarray
            d * n matrix
        b : numpy.ndarray
            d * 1 vector
        lambda1 : float
            Regularization parameter
        lambda2 : float
            Regularization parameter
        opts : dict
            Dictionary containing optimization options:
            loss       : 'l1' (default): loss(e) = ||e||_1 
                         'l2': loss(e) = 0.5 * ||e||_2^2
            tol        : termination tolerance
            max_iter   : maximum number of iterations
            mu         : stepsize for dual variable updating in ADMM
            max_mu     : maximum stepsize
            rho        : rho >= 1, ratio used to increase mu
            DEBUG      : 0 or 1 for printing debug info

    Returns:
        x : numpy.ndarray
            Solution vector
        e : numpy.ndarray
            Error vector
        obj : float
            Objective function value
        err : float
            Residual
        iter : int
            Number of iterations
    """
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)
    loss = opts.get('loss', 'l1')

    d, n = A.shape
    x = np.zeros(n)
    e = np.zeros(d)
    z = np.zeros_like(x)
    Y1 = np.zeros_like(e)
    Y2 = np.zeros_like(x)

    Atb = A.T @ b
    I = np.eye(n)
    invAtAI = np.linalg.inv(A.T @ A + I)

    # Parameters for "flsa" (from SLEP package)
    tol2 = 1e-10
    max_step = 50
    x0 = np.zeros(n - 1)

    for iter in range(1, max_iter + 1):
        xk, ek, zk = x.copy(), e.copy(), z.copy()
        # First super block {x,e}
        x = flsa(x, z - Y2 / mu, x0, lambda1 / mu, lambda2 / mu, n, max_step, tol2, 1, 6)
        if loss == 'l1':
            e = prox_l1(b - A @ z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('Not supported loss function')

        # Second super block {z}
        z = invAtAI @ (-A.T @ (Y1 / mu + e) + Atb + Y2 / mu + x)

        # Compute residuals and errors
        dY1 = A @ z + e - b
        dY2 = x - z
        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgz = np.max(np.abs(zk - z))
        chg = max(chgx, chge, chgz, np.max(np.abs(dY1)), np.max(np.abs(dY2)))

        if DEBUG and (iter == 1 or iter % 10 == 0):
            obj = comp_loss(e, loss) + comp_fusedl1(x, lambda1, lambda2)
            err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
            print(f"iter {iter}, mu={mu}, obj={obj}, err={err}")

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(e, loss) + comp_fusedl1(x, lambda1, lambda2)
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return x, e, obj, err, iter

# 设置参数
opts = {
    'tol': 1e-6,
    'max_iter': 1000,
    'mu': 1e-4,
    'max_mu': 1e10,
    'rho': 1.1,
    'DEBUG': 0,
    'loss': 'l1'
}

# 生成玩具数据
d = 10
na = 200
nb = 100

A = np.random.randn(d, na)
X_true = np.random.randn(na, nb)
B = A @ X_true
b = B[:, 0]

# regularized fused Lasso
lambda1 = 10
lambda2 = 10
X, E, obj, err, iter = fusedl1R(A, b, lambda1, lambda2, opts)
print(f"Final Iteration: {iter}, Objective: {obj}, Error: {err}")
plt.stem(X)
plt.title('Regularized Fused Lasso Result')
plt.show()
