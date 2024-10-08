import json
import multiprocessing
from typing import Collection, Any
import http.client
from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
import admm_utils


class LLMAPI(sampler.LLM):
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, samples_per_prompt: int):
        super().__init__(samples_per_prompt)
        additional_prompt = ('Please modify the ADMM optimizer in the provided igc function to incorporate an unfixed penalty parameter. Specifically, the penalty parameter rho should be adjusted dynamically during each iteration based on the current optimization state, rather than being fixed. Ensure that the updated rho is used correctly in the updates for X, Z, Y1, and Y2.'
                             'Only output the Python code, no descriptions.')
        self._additional_prompt = additional_prompt

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content, self._additional_prompt])
        while True:
            try:
                conn = http.client.HTTPSConnection("api.deepseek.com")

                payload = json.dumps({
                    "messages": [
                        {
                            "content": "You are a helpful programmer who is familar with python and Admm in optimization. ",
                            "role": "system"
                        },
                        {
                            "content": prompt,
                            "role": "user"
                        }
                    ],
                    "model": "deepseek-coder",
                    "frequency_penalty": 0,
                    "max_tokens": 2048,
                    "presence_penalty": 0,
                    "stop": None,
                    "stream": False,
                    "temperature": 1,
                    "top_p": 1,
                    "logprobs": False,
                    "top_logprobs": None
                })
                headers = {
                    'Authorization': 'Bearer sk-66575172e83e40b2bbcaa1cf6b9f0ae8',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                }
                conn.request("POST", "/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                print(data)
                response = data['choices'][0]['message']['content']
                return response
            except Exception:
                continue


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code. Implemented by RZ.

    RZ: Sandbox returns the 'score' of the program and:
    1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Args:
            verbose         : Print evaluate information.
            numba_accelerate: Use numba to accelerate the evaluation. It should be noted that not all numpy functions
                              support numba acceleration, such as np.piecewise().
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            inputs: Any,  # refers to the dataset
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            **kwargs  # RZ: add this
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.

        RZ: If the generated code (generated by LLM) is executed successfully,
        the output of this function is the score of a given program.
        RZ: PLEASE NOTE THAT this SandBox is only designed for bin-packing problem.
        """
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            # if the process is not finished in time, we consider the program illegal
            process.terminate()
            process.join()
            results = None, False
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()
            else:
                results = None, False

        if self._verbose:
            print(f'================= Evaluated Program =================')
            program_: code_manipulation.Program = code_manipulation.text_to_program(text=program)
            func_to_evolve_: str = kwargs.get('func_to_evolve', 'priority')
            function_: code_manipulation.Function = program_.get_function(func_to_evolve_)
            function_: str = str(function_).strip('\n')
            print(f'{function_}')
            print(f'-----------------------------------------------------')
            print(f'Score: {str(results)}')
            print(f'=====================================================')
            print(f'\n\n')

        return results

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_run'
            function_to_run = all_globals_namespace[function_to_run]
            # return the execution results
            results = function_to_run(dataset)
            # the results must be int or float
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except:
            # if raise any exception, we assume the execution failed
            result_queue.put((None, False))

specification = r'''
import numpy as np

# Proximal operator for nuclear norm
def prox_nuclear(B, lambd):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S = np.maximum(S - lambd, 0)
    svp = np.sum(S > 0)
    if svp >= 1:
        S = S[:svp]
        X = U[:, :svp] @ np.diag(S) @ Vt[:svp, :]
        nuclearnorm = np.sum(S)
    else:
        X = np.zeros_like(B)
        nuclearnorm = 0
    return X, nuclearnorm

# Proximal operator for L1 norm
def prox_l1(b, lambd):
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

# Project a point onto a box
def project_box(b, l, u):
    return np.maximum(l, np.minimum(b, u))

# Improved Graph Clustering (IGC) function
@funsearch.evolve
def igc(A, C, lambd, opts):
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    C = np.abs(C)
    d, n = A.shape

    L = np.zeros((d, n))
    S = np.zeros_like(L)
    Z = np.zeros_like(L)
    Y1 = np.zeros_like(L)
    Y2 = np.zeros_like(L)

    for iter in range(max_iter):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()

        # First super block {L, S}
        L, nuclearnormL = prox_nuclear(Z - Y2 / mu, 1 / mu)
        S = prox_l1(-Z + A - Y1 / mu, C * (lambd / mu))

        # Second super block {Z}
        Z = project_box((-S + A + L + (Y2 - Y1) / mu) / 2, 0, 1)

        dY1 = Z + S - A
        dY2 = L - Z
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgL, chgS, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iter == 0 or iter % 10 == 0:
                obj = nuclearnormL + lambd * np.sum(C * np.abs(S))
                err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)
                print(f'iter {iter + 1}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 += mu * dY1
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormL + lambd * np.sum(C * np.abs(S))
    err = np.sqrt(np.linalg.norm(dY1, 'fro')**2 + np.linalg.norm(dY2, 'fro')**2)

    return L, S, obj, err, iter

@funsearch.run
def evaluate(instances: dict) -> float:
    # Generate toy data
    n = instances['n']
    r = instances['r']
    X = np.random.randn(n, r) @ np.random.randn(r, n)
    C = np.random.rand(*X.shape)
    lambda_ = instances['lambda_']
    # Options for the IGC minimization
    opts = instances
    # Perform IGC minimization
    L, S, obj, err, iter = igc(X, C, lambda_, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter
'''

# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4)
    admm_igc_config = admm_utils.datasets['igc']
    global_max_sample_num = 10
    # if it is set to None, funsearch will execute an endless loop
    funsearch.main(
        specification=specification,
        inputs=admm_igc_config,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/low_rank_matrix_models/igc/funsearch_igc',
        temperature=0
    )