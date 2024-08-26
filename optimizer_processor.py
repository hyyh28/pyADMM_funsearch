def process_optimizer_code(code: str, problem_func: str, optimizer_func: str) -> str:
    # Modify the optimizer code with funsearch decorators
    lines = code.splitlines()
    modified_lines = []

    for line in lines:
        if f"def {problem_func}" in line:
            modified_lines.append(f"@funsearch.run")
        elif f"def {optimizer_func}" in line:
            modified_lines.append(f"@funsearch.evolve")
        modified_lines.append(line)

    return "\n".join(modified_lines)
