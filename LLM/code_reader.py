class CodeReader:
    def __init__(self):
        self.code = None
        self.code_loc = None
        self.problem_func = None
        self.optimizer_func = None

    def read_code_from_string(self, code: str):
        if not isinstance(code, str):
            raise TypeError("The code must be a string.")
        if not code.strip():
            raise ValueError("The code string cannot be empty.")
        self.code = code

    def read_code_from_path(self, path: str):
        if not isinstance(path, str):
            raise TypeError("The path must be a string.")
        try:
            with open(path, "r") as f:
                self.code = f.read()
                self.code_loc = f.tell()
        except FileNotFoundError:
            raise FileNotFoundError(f"File at path '{path}' was not found.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file: {e}")

    def _set_problem_func(self, problem_func: str):
        if not isinstance(problem_func, str):
            raise TypeError("The problem function name must be a string.")
        if not problem_func.strip():
            raise ValueError("The problem function name cannot be empty.")
        self.problem_func = problem_func

    def _set_optimizer_func(self, optimizer_func: str):
        if not isinstance(optimizer_func, str):
            raise TypeError("The optimizer function name must be a string.")
        if not optimizer_func.strip():
            raise ValueError("The optimizer function name cannot be empty.")
        self.optimizer_func = optimizer_func

    def _process_code(self):
        if not self.code:
            raise ValueError("No code has been loaded to process.")
        if not self.problem_func or not self.optimizer_func:
            raise ValueError("Problem and optimizer functions must be set before processing code.")

        if f"def {self.problem_func}" not in self.code:
            raise ValueError(f"The problem function '{self.problem_func}' was not found in the code.")
        if f"def {self.optimizer_func}" not in self.code:
            raise ValueError(f"The optimizer function '{self.optimizer_func}' was not found in the code.")

        lines = self.code.splitlines()
        modified_lines = []
        for line in lines:
            if f"def {self.problem_func}" in line:
                modified_lines.append(f"@funsearch.run")
            elif f"def {self.optimizer_func}" in line:
                modified_lines.append(f"@funsearch.evolve")
            modified_lines.append(line)
        new_code = "\n".join(modified_lines)
        self.code = new_code
        return new_code

    def set_problems_and_optimizers(self, problem_func: str, optimizer_func: str):
        self._set_problem_func(problem_func)
        self._set_optimizer_func(optimizer_func)
        return self._process_code()
