from time import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.linalg import lstsq, cholesky
from numpy.linalg import solve
from scipy import sparse
from sklearn.linear_model import Lasso as skLasso

import basis_pursuit


class Lasso(basis_pursuit._ADMM):
    """
    lasso  Solve lasso problem via ADMM

    Solves the following problem via ADMM:

    minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1

    PORTED FROM https://web.stanford.edu/~boyd/admm.html

    The solution is returned in the vector x.

    history is a structure that contains the objective value, the primal and
    dual residual norms, and the tolerances for the primal and dual residual
    norms at each iteration.

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    """

    def __init__(self, lam: float, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4,
                 reltol=1e-2):
        """

        :param rho: augmented lagrangian parameter
        :param.alpha: typical values betwen 1.0 and 1.8
        """
        self.lam = lam
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, A, b):
        x, history = _fit(A, b, self.lam, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x

@jit(cache=True)
def _fit(A, b, lam, rho, alpha, abstol, reltol, max_iter):
    n, p = A.shape
    x = np.zeros((p, 1))
    z = np.zeros((p, 1))
    w = np.zeros((p, 1))
    z_old = np.zeros((p, 1))
    history = np.zeros((5, max_iter))

    L, U = _factor(A, rho)
    Atb = A.T @ b

    for k in range(max_iter):
        # x - update
        q = Atb + rho * (z - w)
        if n >= p:
            x = solve(U, solve(L, q))
        else:
            x = q / rho - A.T @ solve(U, solve(L, A @ q)) / rho ** 2

        # z-update with relaxation
        z_old = z.copy()
        x_hat = alpha * x + (1 - alpha) * z_old
        z = _shrinkage(x_hat + w, lam / rho)

        # u - update
        w = w + (x - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(A, b, lam, x, z)
        history[1, k] = np.linalg.norm(x - z)
        history[2, k] = np.linalg.norm(-rho * (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(x), np.linalg.norm(-z)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(A, b, lam, x, z):
    return 1 / 2 * np.sum((A @ x - b) ** 2) + lam * np.linalg.norm(z, ord=1)


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


@jit(nopython=True, cache=True)
def _factor(A, rho):
    """

    :param A:
    :param kappa:
    """
    n, p = A.shape
    if n >= p:

        L = cholesky(A.T.dot(A) + rho * np.eye(p))
    else:
        L = cholesky(np.eye(n) + 1 / rho * (A @ A.T))
    #L = sparse.csc_matrix(L)
    #U = sparse.csc_matrix(L.T)
    return np.asarray(L), np.asarray(L.T)


def main():
    n = 150
    p = 500
    sparsity = 0.05
    x = sparse.rand(p, 1, sparsity)
    A = np.random.rand(n, p)
    A = A @ sparse.spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x

    lam = 0.1
    rho = 0.05

    x_true = x.toarray()

    lasso = Lasso(lam=0.1, rho=0.05, alpha=1, max_iter=100)
    x = lasso.fit(A, b)
    t0 = time()
    x = lasso.fit(A, b)
    print(time() - t0)

    sklasso = skLasso(alpha=lam)
    t0 = time()
    x = sklasso.fit(A, b)
    print(time() - t0)

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot(lasso.history['objval'])
    axs[0].set_ylabel('f(x^k) + g(z^k)')

    axs[1].plot(lasso.history['r_norm'])
    axs[1].plot(lasso.history['eps_pri'])
    axs[1].set_yscale('log')
    axs[1].set_ylabel('||r||_2')

    axs[2].plot(lasso.history['s_norm'])
    axs[2].plot(lasso.history['eps_dual'])
    axs[2].set_yscale('log')
    axs[2].set_ylabel('||s||_2')
    axs[2].set_xlabel('iter (k)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
