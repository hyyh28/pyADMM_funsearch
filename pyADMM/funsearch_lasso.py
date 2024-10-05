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


class Lasso(basis_pursuit._ADMM):
    def __init__(self, lam: float, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4,
                 reltol=1e-2):
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
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


@jit(nopython=True, cache=True)
def _factor(A, rho):
    n, p = A.shape
    if n >= p:
        L = cholesky(A.T.dot(A) + rho * np.eye(p))
    else:
        L = cholesky(np.eye(n) + 1 / rho * (A @ A.T))
    return np.asarray(L), np.asarray(L.T)


def get_llm_suggestion(previous_reward=None):
    prompt = "Suggest parameters lam, rho, and alpha for the ADMM optimizer to deal with LASSO problem. Only output the parameters such like lam=x, rho=y, alpha=z,(where x, y, z are float) no descriptions."
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

    # performance = np.mean((x_true.ravel() - x_pred.ravel()) ** 2)
    iterations = len(lasso.history['objval'])  # Number of iterations
    performance = lasso.history['objval'][iterations-1]

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
