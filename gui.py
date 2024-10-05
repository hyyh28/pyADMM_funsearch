import tkinter as tk
from tkinter import scrolledtext, messagebox
from optimizer_processor import process_optimizer_code
from llm_services import LLMCopilot, LLMHyperparameterOptimizer, LLMCodeGenerator


class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimization Problem Processor")

        # Input Text Area for the optimizer code
        self.code_label = tk.Label(root, text="Enter your ADMM optimizer code:")
        self.code_label.pack()
        self.code_input = scrolledtext.ScrolledText(root, width=80, height=20)
        self.code_input.pack()

        # Entry for optimization problem function
        self.problem_func_label = tk.Label(root, text="Optimization Problem Function:")
        self.problem_func_label.pack()
        self.problem_func_entry = tk.Entry(root)
        self.problem_func_entry.pack()

        # Entry for optimizer function
        self.optimizer_func_label = tk.Label(root, text="Optimizer Function:")
        self.optimizer_func_label.pack()
        self.optimizer_func_entry = tk.Entry(root)
        self.optimizer_func_entry.pack()

        # Button to process the code
        self.process_button = tk.Button(root, text="Process Code", command=self.process_code)
        self.process_button.pack()

        # Output area for the modified code
        self.modified_code_label = tk.Label(root, text="Modified Code with Decorators:")
        self.modified_code_label.pack()
        self.modified_code_output = scrolledtext.ScrolledText(root, width=80, height=20)
        self.modified_code_output.pack()

        # Buttons for LLM services
        self.copilot_button = tk.Button(root, text="LLM Copilot Advice", command=self.run_copilot)
        self.copilot_button.pack()

        self.hyperopt_button = tk.Button(root, text="LLM Hyperparameter Optimization", command=self.run_hyperopt)
        self.hyperopt_button.pack()

        self.codegen_button = tk.Button(root, text="LLM Code Generation", command=self.run_codegen)
        self.codegen_button.pack()

    def process_code(self):
        code = self.code_input.get("1.0", tk.END)
        problem_func = self.problem_func_entry.get()
        optimizer_func = self.optimizer_func_entry.get()

        if not code.strip() or not problem_func or not optimizer_func:
            messagebox.showerror("Input Error", "Please fill in all fields.")
            return

        modified_code = process_optimizer_code(
            code=code,
            problem_func=problem_func,
            optimizer_func=optimizer_func
        )
        self.modified_code_output.delete("1.0", tk.END)
        self.modified_code_output.insert(tk.END, modified_code)

    def run_copilot(self):
        code = self.modified_code_output.get("1.0", tk.END)
        copilot = LLMCopilot()
        copilot.provide_advice(code)

    def run_hyperopt(self):
        code = self.modified_code_output.get("1.0", tk.END)
        optimizer = LLMHyperparameterOptimizer()
        optimizer.optimize_hyperparameters(code)

    def run_codegen(self):
        code = self.modified_code_output.get("1.0", tk.END)
        code_generator = LLMCodeGenerator()
        code_generator.improve_code(code)


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()
