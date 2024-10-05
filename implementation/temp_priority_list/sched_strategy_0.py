
import numpy as np
def priority(parameters, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, step):
    """
    Update parameters using the Adam optimization algorithm.

    Parameters:
    parameters (iterable of tensors): The parameters to be updated.
    grads (iterable of tensors): The gradients of the parameters.
    exp_avgs (iterable of tensors): Exponential moving averages of gradients.
    exp_avg_sqs (iterable of tensors): Exponential moving averages of squared gradients.
    lr (float): Learning rate.
    beta1 (float): Exponential decay rate for the first moment estimates.
    beta2 (float): Exponential decay rate for the second moment estimates.
    eps (float): Small constant to prevent division by zero.
    step (int): Current optimization step.

    Returns:
    None
    """
    for param, grad, exp_avg, exp_avg_sq in zip(parameters, grads, exp_avgs, exp_avg_sqs):
        if grad is None:
            continue

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

        param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)