from LLM.gpt import DeepSeekClient
class LLMCopilot:
    def __init__(self, client: DeepSeekClient):
        self.client = client

    def get_advice(self, user_query: str) -> str:
        """
        Allows the user to ask for advice directly from the LLM.
        """
        prompt = f"You are an experienced mathematician and optimization expert. {user_query}"
        response = self.client.get_llm_suggestion(prompt)
        return response


class HyperParameterSelector:
    def __init__(self, client: DeepSeekClient, num_iterations: int = 10):
        self.client = client
        self.num_iterations = num_iterations
        self.copilot = LLMCopilot(client)
        self.history = {
            'lams': [],
            'rhos': [],
            'alphas': [],
            'rewards': []
        }

    def get_suggestions(self, previous_reward=None):
        prompt = "Suggest parameters lam, rho, and alpha for the ADMM optimizer to deal with LASSO problem. Only output the parameters such as lam=x, rho=y, alpha=z (where x, y, z are float) without any descriptions."
        if previous_reward is not None:
            prompt += f" The previous reward was {previous_reward:.4f}. You need to improve the performance by giving better parameters."

        suggestion = self.client.get_llm_suggestion(prompt)
        if suggestion:
            params = {}
            for param in suggestion.split(","):
                key, value = param.split("=")
                params[key.strip()] = float(value.strip())
            return params['lam'], params['rho'], params['alpha']
        else:
            return None, None, None

    def evaluate_parameters(self, lam, rho, alpha, A, b, x_true):
        lasso = Lasso(lam=lam, rho=rho, alpha=alpha, max_iter=100)
        t0 = time()
        x_pred = lasso.fit(A, b)
        elapsed_time = time() - t0

        iterations = len(lasso.history['objval'])
        performance = lasso.history['objval'][iterations - 1]

        return elapsed_time, performance, iterations

    def calculate_reward(self, performance):
        return -performance

    def optimize(self, A, b, x_true):
        previous_reward = None

        for i in range(self.num_iterations):
            lam, rho, alpha = self.get_suggestions(previous_reward)
            if lam is None or rho is None or alpha is None:
                print(f"Iteration {i + 1}: Failed to get valid parameters.")
                continue

            time_taken, performance, iterations = self.evaluate_parameters(lam, rho, alpha, A, b, x_true)
            reward = self.calculate_reward(time_taken)

            print(
                f"Iteration {i + 1}: lam={lam}, rho={rho}, alpha={alpha}, time={time_taken:.4f}s, performance={performance:.4f}, iterations={iterations}, reward={reward:.4f}")

            self.history['lams'].append(lam)
            self.history['rhos'].append(rho)
            self.history['alphas'].append(alpha)
            self.history['rewards'].append(reward)

            previous_reward = reward

        return self.history

    def plot_learning_curves(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(self.history['lams'], marker='o')
        axs[0].set_ylabel('lam')
        axs[0].set_title('Learning Curves for Lasso')

        axs[1].plot(self.history['rhos'], marker='o')
        axs[1].set_ylabel('rho')

        axs[2].plot(self.history['alphas'], marker='o')
        axs[2].set_ylabel('alpha')

        axs[3].plot(self.history['rewards'], marker='o')
        axs[3].set_ylabel('reward')
        axs[3].set_xlabel('Iteration')

        plt.tight_layout()
        plt.show()

    def ask_copilot(self, query: str):
        """
        Allows the user to interact with the LLM Copilot to get advice on designing hyper-parameters.
        """
        return self.copilot.get_advice(query)


# Example usage in main function
def main():
    n = 150
    p = 500
    sparsity = 0.05
    x = sparse.rand(p, 1, sparsity)
    A = np.random.rand(n, p)
    A = A @ sparse.spdiags(1 / np.sqrt(np.sum(A ** 2, axis=0)), 0, p, p)
    b = A @ x

    x_true = x.toarray()

    # Assuming the DeepSeekClient class is already defined and implemented
    client = DeepSeekClient(api_key="your_openai_api_key")
    selector = HyperParameterSelector(client)

    # Interact with the Copilot
    user_query = "What are the best practices for selecting hyper-parameters in LASSO problems?"
    advice = selector.ask_copilot(user_query)
    print(f"LLM Copilot Advice: {advice}")

    selector.optimize(A, b, x_true)
    selector.plot_learning_curves()


if __name__ == "__main__":
    main()
