import openai
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from sklearn.linear_model import Lasso
from time import time

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Generate synthetic data
n = 150
p = 500
sparsity = 0.05
x_true = sparse.rand(p, 1, sparsity)
A = np.random.rand(n, p)
A = A @ sparse.spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
b = A @ x_true


def get_llm_suggestion(previous_reward=None):
    prompt = "Suggest parameters lam, rho, and alpha for the Lasso ADMM optimizer."
    if previous_reward is not None:
        prompt += f" The previous reward was {previous_reward:.4f}."

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    suggestion = response.choices[0].text.strip()
    # Parse the suggestion assuming it is a string like "lam=0.1, rho=0.05, alpha=1"
    params = {}
    for param in suggestion.split(","):
        key, value = param.split("=")
        params[key.strip()] = float(value.strip())

    return params['lam'], params['rho'], params['alpha']


def evaluate_parameters(lam, rho, alpha):
    lasso = Lasso(alpha=lam, max_iter=100)
    t0 = time()
    lasso.fit(A, b.toarray().ravel())
    elapsed_time = time() - t0
    x_pred = lasso.coef_

    # Compute performance: e.g., mean squared error
    performance = np.mean((x_true.toarray().ravel() - x_pred) ** 2)

    return elapsed_time, performance


def calculate_reward(time_taken, performance):
    # Simple reward function combining speed and performance
    return 1 / (time_taken * performance)


# Main optimization loop
num_iterations = 10
previous_reward = None

for i in range(num_iterations):
    lam, rho, alpha = get_llm_suggestion(previous_reward)
    time_taken, performance = evaluate_parameters(lam, rho, alpha)
    reward = calculate_reward(time_taken, performance)

    print(
        f"Iteration {i + 1}: lam={lam}, rho={rho}, alpha={alpha}, time={time_taken:.4f}s, performance={performance:.4f}, reward={reward:.4f}")

    previous_reward = reward
