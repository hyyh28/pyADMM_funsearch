# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile

from implementation import code_manipulation
from implementation import programs_database


def _trim_preface_of_body(sample: str) -> str:
    """Implemented by RZ: Trim the redundant descriptions/symbols/'def' declaration before the function body.
    Example sample:
    -------------------------------------
    This is the optimized function ...
    def adam_update_v2(...) -> ...:
        return ...
    -------------------------------------
    Example return of this function:
    -------------------------------------
    return ...
    -------------------------------------
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        # find the first 'def' statement in the given code
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extracts the body of the generated function, trimming anything after it.

    RZ: the arg generated_code must only include the body of the generated function (an example is shown below):
    --------------
        a = item
        return a
    --------------
    Please note that the indentation is REQUIRED !!! I don't know why they write code like this !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    # We keep trying and deleting code from the end until the parser succeeds.
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # RZ: "e.lineno - 1" locates the line number of the lost python code
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        # Nothing could be saved from `generated_code`
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """Returns the compiled generated function and the full runnable program.
    RZ: This function removes the code after the generated function body.
    """
    generated_code = _trim_preface_of_body(generated_code)
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )
    if body == '':
        body = 'a'
    
    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


class Sandbox(ABC):
    """Sandbox for executing generated code.
    RZ: Sandbox 1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  # refers to the full dataset, added by RZ
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.
        RZ: If the generated code (generated by LLM) is executed successfully, the output of this function
        """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Returns whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        # In `program` passed into this function the most recently generated
        # function has already been renamed to `function_to_evolve` (wihout the
        # suffix). Therefore, any function call starting with `function_to_evolve_v`
        # is a call to an ancestor function.
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False


class Evaluator:
    """Class that analyses functions generated by LLMs."""

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            template: code_manipulation.Program,
            function_to_evolve: str,  # RZ: refers to the name of the function to evolve (e.g., 'priority')
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            inputs: Sequence[Any],  # RZ: I guess this refers to the evaluate instance
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()
    def extract_priority_body(self, code):
        class FunctionBodyExtractor(ast.NodeVisitor):
            def __init__(self):
                self.function_body_code = ""
                
            def visit_FunctionDef(self, node):
                if node.name == 'adam_update':
                    # Start with an empty list to hold code lines
                    code_lines = []
                    # Iterate over each statement in the function body
                    for stmt in node.body:
                        # Skip docstrings
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str):
                            continue
                        # Unparse the statement, add indentation, and append a newline
                        code_line = "    " + ast.unparse(stmt).replace('\n', '\n    ') + "\n"
                        code_lines.append(code_line)
                    # Join all lines into a single string
                    self.function_body_code = "".join(code_lines)
    
        # Parse the source code into an AST
        tree = ast.parse(code)
        extractor = FunctionBodyExtractor()
        extractor.visit(tree)
        
        return extractor.function_body_code
    # def update_priority_function_body(self, new_body_code):
    #     import ast
    #     import textwrap
    #     import os
    #     import re
    #     file_path = './implementation/adam/adam_pri.py'
    #     # with open(file_path, 'r') as file:
    #     #     lines = file.readlines()

    #     # # Find the start and end of the priority function's body
    #     # start_index, end_index = None, None
    #     # for i, line in enumerate(lines):
    #     #     if line.strip().startswith("def priority("):
    #     #         start_index = i + 1
    #     #         while not lines[start_index].strip():
    #     #             start_index += 1  # Skip empty lines or comments before the function body
    #     #     if start_index and line.strip() == "":
    #     #         end_index = i
    #     #         break

    #     # # Replace the function body
    #     # if start_index is not None and end_index is not None:
    #     #     # Prepare the new function body with correct indentation
    #     #     new_body = ['    ' + line + '\n' for line in new_body_code.split('\n')]
    #     #     lines[start_index:end_index] = new_body
    #     # else:
    #     #     raise ValueError("Function 'priority' not found in the file.")
    #     # import pdb;pdb.set_trace()
    #     # # Write the modified contents back to the file
    #     # with open(file_path, 'w') as file:
    #     #     file.writelines(lines)
    #     with open(file_path, 'r') as file:
    #         content = file.read()
    
    #     # Pattern to find the priority function's body
    #     pattern = r"(def priority\([^\)]*\):.*?\n)([\s\S]*?)(\n\n|\Z)"
        
    #     # Replacement string with new_body_code
    #     replacement = r"\g<1>" + "    " + new_body_code.replace("\n", "\n    ") + r"\3"

    #     # Replace the function body
    #     updated_content = re.sub(pattern, replacement, content, count=1)
        
    #     # Write the updated content back to the file
    #     with open(file_path, 'w') as file:
    #         file.write(updated_content)

    class ReplaceFunctionBody(ast.NodeTransformer):
        def __init__(self, function_name, new_body_ast):
            self.function_name = function_name
            self.new_body_ast = ast.parse(new_body_ast).body  # 将新函数体代码解析为AST

        def visit_FunctionDef(self, node):
            # 查找目标函数并替换其函数体
            if node.name == self.function_name:
                node.body = self.new_body_ast
                return node
            return self.generic_visit(node)  # 处理其他所有节点

    def update_priority_function_body(self, new_body_code):
        # 读取原Python文件
        file_path = './implementation/adam/adam_pri.py'
        with open(file_path, 'r', encoding='utf-8') as file:
            original_code = file.read()

        # 解析原代码为AST
        original_ast = ast.parse(original_code)

        # 应用AST转换器
        transformer = ReplaceFunctionBody(function_name, new_body_code)
        new_ast = transformer.visit(original_ast)

        # 将修改后的AST转换回源代码
        new_code = astor.to_source(new_ast)
        import pdb;pdb.set_trace()
        # 将新代码写回原文件或新文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_code)
    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs  # RZ: add this to do profile
    ) -> None:
        """Compiles the sample into a program and executes it on test inputs.

        Args:
            sample: RZ: please note that the sample must be preprocessed--only have function body,
                    no description before it (except annotations), no symbols before it.
                    Or the "_sample_to_program" function will fail!!!
        """
        # RZ: 'new_function' refers to the evolved function ('def' statement + function body)
        # RZ: 'program' is the template code + new_function
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        
        scores_per_test = {}
        try:
            self.update_priority_function_body(self.extract_priority_body(program))
        except SyntaxError as e:
            runs_ok = False
        except Exception as e:
            runs_ok = False

        time_reset = time.time()
        for current_input in self._inputs:
            # RZ: IMPORTANT !!! if self._inputs is a dict,
            # current_input is a key (perhaps in string type)
            # do not ignore this when implementing SandBox !!!
            
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            
            # if 'return' not in str(new_function):
            #     runs_ok = False
            # import pdb;pdb.set_trace()
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if not isinstance(test_output, (int, float)):
                    print(f'RZ=> Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset

        # RZ: If 'score_per_test' is not empty, the score of the program will be recorded to the profiler by the 'register_program'.
        # This is because the register_program will do reduction for a given Function score.
        # If 'score_per_test' is empty, we record it to the profiler at once.
        
        if scores_per_test:
            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
            
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)
