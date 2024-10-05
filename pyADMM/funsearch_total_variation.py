from time import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky, solve
from numba import jit
import basis_pursuit
from openai import OpenAI



class TotalVariation(basis_pursuit._ADMM):
    def __init__(self, lam: float, rho: float, alpha: float, quiet: bool = True, max_iter=1000, abstol=1e-4,
                 reltol=1e-2):
        self.lam = lam
        super().__init__(rho, alpha, quiet, max_iter, abstol, reltol)

    def fit(self, b):
        x, history = _fit(b, self.lam, self.rho, self.alpha, self.abstol, self.reltol, self.max_iter)
        self.history = {
            'objval': history[0],
            'r_norm': history[1],
            's_norm': history[2],
            'eps_pri': history[3],
            'eps_dual': history[4]
        }
        return x


@jit(cache=True)
def _fit(b, lam, rho, alpha, abstol, reltol, max_iter):
    n = b.shape[0]

    e = np.ones(n)
    D = np.diag(e) - np.diag(e, 1)[:n, :n]
    x = np.zeros((n, 1))
    z = np.zeros((n, 1))
    w = np.zeros((n, 1))

    I = np.eye(n)
    DtD = D.T @ D

    history = np.zeros((5, max_iter))

    for k in range(max_iter):
        # x - update
        x = solve(I + rho * DtD, b + rho * D.T @ (z - w))

        # z-update with relaxation
        z_old = z.copy()
        Ax_hat = alpha * D @ x + (1 - alpha) * z_old
        z = _shrinkage(Ax_hat + w, lam / rho)

        # u - update
        w = w + (Ax_hat - z)

        # diagnostics, reporting, termination checks
        history[0, k] = _objective(b, lam, x, z)
        history[1, k] = np.linalg.norm(D @ x - z)
        history[2, k] = np.linalg.norm(-rho * D.T @ (z - z_old))
        history[3, k] = np.sqrt(n) * abstol + reltol * np.max(np.array([np.linalg.norm(D @ x), np.linalg.norm(-z)]))
        history[4, k] = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * D.T @ w)
        if history[1][k] < history[3][k] and history[2][k] < history[4][k]:
            break
    return x, history


@jit(nopython=True, cache=True)
def _objective(b, lam, x, z):
    return 1 / 2 * np.sum((x - b) ** 2) + lam * np.linalg.norm(z, ord=1)


@jit(nopython=True, cache=True)
def _shrinkage(a, kappa):
    return np.maximum(0, a - kappa) - np.maximum(0, -a - kappa)


def get_llm_suggestion(previous_reward=None):
    prompt = "Suggest parameters lam, rho, and alpha for the ADMM optimizer to deal with minimize total variance problem. Only output the parameters such like lam=x, rho=y, alpha=z,(where x, y, z are float) no descriptions."
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
        temperature=0.6,
        stream=False
    )

    suggestion = response.choices[0].message.content
    params = {}
    for param in suggestion.split(","):
        key, value = param.split("=")
        params[key.strip()] = float(value.strip())

    return params['lam'], params['rho'], params['alpha']


def calculate_reward(performance):
    return performance


def evaluate_parameters(lam, rho, alpha, b, x_true):
    tv = TotalVariation(lam=lam, rho=rho, alpha=alpha, max_iter=100)
    t0 = time()
    x_pred = tv.fit(b)
    elapsed_time = time() - t0

    performance = np.mean((x_true.ravel() - x_pred.ravel()) ** 2)
    iterations = len(tv.history['objval'])

    return elapsed_time, performance, iterations


def main():
    n = 100
    x = np.ones((n, 1))
    for _ in range(3):
        idx = np.random.randint(n, size=1)
        k = np.random.randint(10, size=1)
        x[int(idx / 2):int(idx)] = k * x[int(idx / 2):int(idx)]

    x_true = x
    b = x + np.random.randn(n, 1)

    num_iterations = 10
    previous_reward = None

    lams = []
    rhos = []
    alphas = []
    rewards = []

    for i in range(num_iterations):
        lam, rho, alpha = get_llm_suggestion(previous_reward)
        time_taken, performance, iterations = evaluate_parameters(lam, rho, alpha, b, x_true)
        reward = calculate_reward(performance)

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
    axs[0].set_title('Learning Curves for Minimize total Variance')

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
