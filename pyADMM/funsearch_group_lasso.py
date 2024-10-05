from time import time
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy.linalg import lstsq, cholesky, solve
from scipy import sparse
from sklearn.linear_model import Lasso as skLasso
import basis_pursuit
from openai import OpenAI

# Set your OpenAI API key
# openai.api_key = 'your_openai_api_key'


class GroupLasso(basis_pursuit._ADMM):
    """
    group_lasso  Solve group lasso problem via ADMM

    solves the following problem via ADMM:

    minimize 1/2*|| Ax - b ||_2^2 + \lambda sum(norm(x_i))

    PORTED FROM https://web.stanford.edu/~boyd/admm.html

    The input p is a K-element vector giving the block sizes n_i, so that x_i
    is in R^{n_i}.

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

    def fit(self, A, b, p):
        x, history = _fit(A, b, p, self.lam, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


# @jit(nopython=True)
def _fit(A, b, partition, lam, rho, alpha, abstol, reltol, max_iter):
    n, p = A.shape
    Atb = A.T @ b
    x = np.zeros((p, 1))
    z = np.zeros((p, 1))
    w = np.zeros((p, 1))

    cum_part = np.cumsum(partition)

    history = np.zeros((5, max_iter))

    L,U=_factor(A,rho)

    for k in range(max_iter):

        q = Atb + rho * (z - w)
        if n >= p:
            x = solve(U, solve(L, q))
        else:
            x = q / rho - A.T @ solve(U, solve(L, A @ q)) / rho ** 2

        # z-update with relaxation
        z_old = z
        x_hat = alpha * x + (1 - alpha) * z_old
        for i, start_ind in enumerate(cum_part[:-1]):
            z[start_ind:cum_part[i + 1]] = _shrinkage(x_hat[start_ind:cum_part[i + 1]] + w[start_ind:cum_part[i + 1]], lam / rho)

        # u - update
        w = w + (x - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(A, b, lam, cum_part, x, z)
        history[1, k] = np.linalg.norm(x - z)
        history[2, k] = np.linalg.norm(-rho * (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(x), np.linalg.norm(-z)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


# @jit(nopython=True)
def _objective(A, b, lam, cum_part, x, z):
    obj = sum(
        np.linalg.norm(z[start_ind : cum_part[i + 1]])
        for i, start_ind in enumerate(cum_part[:-1])
    )

    return (1 / 2 * np.sum((A @ x - b) ** 2) + lam * obj)


# @jit(nopython=True)
def _shrinkage(a, kappa):
    """

    :param a:
    :param kappa:
    """
    return a * np.maximum(np.linalg.norm(a) - kappa, 0.)/np.linalg.norm(a)

# @jit(nopython=True, cache=True)
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


def get_llm_suggestion(previous_reward=None):
    prompt = "Suggest parameters lam, rho, and alpha for the ADMM optimizer to deal with Group LASSO problem. Only output the parameters such like lam=x, rho=y, alpha=z,(where x, y, z are float) no descriptions."
    if previous_reward is not None:
        prompt += f" The previous reward was {previous_reward:.4f}. You need to improve the performance by given better parameters."
    client = OpenAI(api_key="sk-66575172e83e40b2bbcaa1cf6b9f0ae8", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an experienced mathematician."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
        stream=False
    )

    suggestion = response.choices[0].message.content
    params = {}
    for param in suggestion.split(","):
        key, value = param.split("=")
        params[key.strip()] = float(value.strip())

    return params['lam'], params['rho'], params['alpha']


def calculate_reward(performance):
    # Reward function: maxmize the number of iterations
    return -performance


def evaluate_parameters(lam, rho, alpha, A, b, x_true):
    lasso = Lasso(lam=lam, rho=rho, alpha=alpha, max_iter=100)
    t0 = time()
    x_pred = lasso.fit(A, b)
    elapsed_time = time() - t0

    performance = np.mean((x_true.ravel() - x_pred.ravel()) ** 2)
    iterations = len(lasso.history['objval'])  # Number of iterations

    return elapsed_time, performance, iterations


def main():
    n = 150
    p = 500
    sparsity = 0.05
    x = sparse.rand(p, 1, sparsity)
    A = np.random.rand(n, p)
    A = A @ sparse.spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x

    x_true = x.toarray()

    num_iterations = 10
    previous_reward = None

    lams = []
    rhos = []
    alphas = []
    rewards = []

    for i in range(num_iterations):
        lam, rho, alpha = get_llm_suggestion(previous_reward)
        time_taken, performance, iterations = evaluate_parameters(lam, rho, alpha, A, b, x_true)
        reward = calculate_reward(time_taken)

        print(
            f"Iteration {i + 1}: lam={lam}, rho={rho}, alpha={alpha}, time={time_taken:.4f}s, performance={performance:.4f}, iterations={iterations}, reward={reward:.4f}")

        lams.append(lam)
        rhos.append(rho)
        alphas.append(alpha)
        rewards.append(reward)

        previous_reward = reward

    # Plotting the learning curves
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(lams, marker='o')
    axs[0].set_ylabel('lam')
    axs[0].set_title('Learning Curves for lasso')

    axs[1].plot(rhos, marker='o')
    axs[1].set_ylabel('rho')

    axs[2].plot(alphas, marker='o')
    axs[2].set_ylabel('alpha')

    axs[3].plot(rewards, marker='o')
    axs[3].set_ylabel('reward')
    axs[3].set_xlabel('Iteration')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
