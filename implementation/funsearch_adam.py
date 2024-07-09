from collections.abc import Sequence
from typing import Any

from implementation import code_manipulation
from implementation import config as config_lib
from implementation import evaluator
from implementation import programs_database
from implementation import sampler

def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]

def _get_call_function(specification: str):
    call_function = code_manipulation.get_functions_called(specification)
    return call_function

def load_adam_code():
    adam_code_file = "../optimizers/adam.py"
    with open(adam_code_file, 'r') as f:
        ff = f.read()
        return ff

if __name__ == "__main__":
    adam_code = load_adam_code()
    # to_evolve, to_run = _extract_function_names(adam_code)
    # function_to_evolve, function_to_run = _extract_function_names(adam_code)
    function_to_evolve = "test"
    function_list = _get_call_function(adam_code)
    template = code_manipulation.text_to_program(adam_code)
    config = config_lib.Config()
    database = programs_database.ProgramsDatabase(
        config.programs_database, template, function_to_evolve)
    evaluators = []
    initial = template.get_function(function_to_evolve).body
    print(initial)
