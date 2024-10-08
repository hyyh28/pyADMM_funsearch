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
        additional_prompt = ('Please modify the ADMM optimizer in the provided rmsc function to incorporate an unfixed penalty parameter. Specifically, the penalty parameter rho should be adjusted dynamically during each iteration based on the current optimization state, rather than being fixed. Ensure that the updated rho is used correctly in the updates for X, Z, Y1, and Y2.'
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
from scipy.linalg import svd

def generate_toy_data(d=10, na=200, nb=100):
    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = np.dot(A, X)
    b = B[:, 0]
    return A, X, B, b

def prox_l1(b, lambd):
    return np.maximum(0, b - lambd) + np.minimum(0, b + lambd)

def prox_nuclear(B, lambd):
    U, S, Vh = svd(B, full_matrices=False)
    S = np.maximum(S - lambd, 0)
    X = np.dot(U * S, Vh)
    nuclearnorm = np.sum(S)
    return X, nuclearnorm

def project_simplex(B):
    n, m = B.shape
    B_sort = np.sort(B, axis=1)[:, ::-1]
    cum_B = np.cumsum(B_sort, axis=1)
    A = np.arange(1, m + 1)
    sigma = B_sort - (cum_B - 1) / A
    idx = np.sum(sigma > 0, axis=1)
    sigma = np.take_along_axis(B_sort, np.expand_dims(idx - 1, axis=1), axis=1)
    sigma = np.tile(sigma, (1, m))
    return np.maximum(B - sigma, 0)

@funsearch.evolve
def rmsc(X, lambd, opts):
    # Set default options
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, n, m = X.shape
    L = np.zeros((d, n))
    S = np.zeros((d, n, m))
    Z = L.copy()
    Y = S.copy()
    Y2 = L.copy()
    
    for iter in range(max_iter):
        Lk, Sk, Zk = L.copy(), S.copy(), Z.copy()
        
        Z, nuclearnormZ = prox_nuclear(L + Y2 / mu, 1 / mu)
        for i in range(m):
            S[:, :, i] = prox_l1(X[:, :, i] - L - Y[:, :, i] / mu, lambd / mu)
        
        L = project_simplex((np.sum(X - S - Y / mu, axis=2) + Z - Y2 / mu) / (m + 1))
        
        dY = np.zeros_like(S)
        for i in range(m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]
        dY2 = L - Z
        
        chgL = np.max(np.abs(Lk - L))
        chgZ = np.max(np.abs(Zk - Z))
        chgS = np.max([np.max(np.abs(Sk[:, :, i] - S[:, :, i])) for i in range(m)])
        chg = max(chgL, chgZ, chgS, np.max(np.abs(dY)), np.max(np.abs(dY2)))
        
        if DEBUG and (iter == 0 or iter % 10 == 0):
            obj = nuclearnormZ + lambd * np.sum(np.abs(S))
            err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
            print(f'iter {iter}, mu={mu}, obj={obj}, err={err}')
        
        if chg < tol:
            break
        
        Y += mu * dY
        Y2 += mu * dY2
        mu = min(rho * mu, max_mu)
    
    obj = nuclearnormZ + lambd * np.sum(np.abs(S))
    err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    
    return L, S, obj, err, iter
    
@funsearch.run
def evaluate(instances:dict) -> float:
    # Options for the rmsc minimization
    opts = instances
    d, n, m = opts['d'], opts['n'], opts['m']
    
    # Generate toy data
    X = np.random.randn(n, n, m)
    lambda_val = 1 / np.sqrt(n)
    
    # Perform rmsc minimization
    L, S, obj, err, iter = rmsc(X, lambda_val, opts)
    print(f'Iterations: {iter}, Objective: {obj}, Error: {err}')
    return -iter
'''

if __name__ == '__main__':
    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4)
    rmsc_config = admm_utils.datasets['rmsc']
    global_max_sample_num = 10
    funsearch.main(
        specification=specification,
        inputs=rmsc_config,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/low_rank_matrix_models/rmsc/funsearch_rmsc',
        temperature=0
    )

