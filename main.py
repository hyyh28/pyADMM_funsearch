from optimizer_processor import process_optimizer_code
from llm_services import LLMCopilot, LLMHyperparameterOptimizer, LLMCodeGenerator


def main():
    # Accept user input
    optimizer_code = input("Please provide your ADMM optimizer code:\n")
    optimization_problem_function = input("Which function represents the optimization problem?\n")
    optimizer_function = input("Which function represents the optimizer?\n")

    # Process the code
    modified_code = process_optimizer_code(
        code=optimizer_code,
        problem_func=optimization_problem_function,
        optimizer_func=optimizer_function
    )

    print("Modified code with funsearch decorators:\n", modified_code)

    # Initialize LLM services
    copilot = LLMCopilot()
    optimizer = LLMHyperparameterOptimizer()
    code_generator = LLMCodeGenerator()

    # Example usage of LLM services
    copilot.provide_advice(modified_code)
    optimizer.optimize_hyperparameters(modified_code)
    code_generator.improve_code(modified_code)


if __name__ == "__main__":
    main()
