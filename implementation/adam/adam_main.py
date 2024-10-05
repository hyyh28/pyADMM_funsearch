

# # 定义目标函数
# def funcVal(X):
#     return 5 * X[0] ** 2 + 2 * X[1] ** 2 + 3 * X[0] - 10 * X[1] + 4

# # 定义目标函数的梯度
# def gradient(X):
#     grad_x1 = 10 * X[0] + 3
#     grad_x2 = 4 * X[1] - 10
#     return np.array([grad_x1, grad_x2])

# Adam优化算法的函数实现

# def adam_optimize(file_name, grad_func, X_init, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=100000, grad_tolerance=1e-6, func_tolerance=1e-6, param_tolerance=1e-6):
#     m = np.zeros_like(X_init)
#     v = np.zeros_like(X_init)
#     t = 0
#     X = X_init.copy()
#     prev_func_val = funcVal(X)
#     import dill
#     with open(file_name, 'r') as file:
#         read_function_code = file.read()
#     local_vars = {}
#     exec(read_function_code, globals(), local_vars)
#     priority = local_vars['priority']
#     for i in range(num_iterations):
#         t += 1
#         grad = grad_func(X)
        
#         # if np.linalg.norm(grad) < grad_tolerance:
#         #     print(f"Gradient norm below tolerance at iteration {i+1}.")
#         #     break
        
#         X_update, m, v = priority(m, v, grad, t, beta1, beta2, lr, epsilon)
        
#         X_new = X + X_update
#         # if np.linalg.norm(X_new - X) < param_tolerance:
#         #     break
        
#         current_func_val = funcVal(X_new)
#         if abs(current_func_val - (-8.949965032491233)) < func_tolerance:
#             break
        
#         X = X_new
#         prev_func_val = current_func_val
        
#     return X, funcVal(X), i



# def adam_update(parameters, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, step):
#     """
#     Update parameters using the Adam optimization algorithm.

#     Parameters:
#     parameters (iterable of tensors): The parameters to be updated.
#     grads (iterable of tensors): The gradients of the parameters.
#     exp_avgs (iterable of tensors): Exponential moving averages of gradients.
#     exp_avg_sqs (iterable of tensors): Exponential moving averages of squared gradients.
#     lr (float): Learning rate.
#     beta1 (float): Exponential decay rate for the first moment estimates.
#     beta2 (float): Exponential decay rate for the second moment estimates.
#     eps (float): Small constant to prevent division by zero.
#     step (int): Current optimization step.

#     Returns:
#     None
#     """
#     """Improved version of `adam_update_v0`."""

#     # Calculate the bias corrections for the moving averages
#     for param, grad, exp_avg, exp_avg_sq in zip(parameters, grads, exp_avgs, exp_avg_sqs):
#         if grad is None:
#             continue

#         exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#         exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#         bias_correction1 = 1 - beta1 ** step
#         bias_correction2 = 1 - beta2 ** step

#         step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

#         param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
# 下载mnist手写数据集
# 训练集

 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_data = torchvision.datasets.MNIST(
#     root='/mnt/data/wjh/temp/test_adam/funs/funsearch_vm_adam/implementation/adam/MNIST/',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=DOWNLOAD_MNIST,
# )

# test_data = torchvision.datasets.MNIST(root='/mnt/data/wjh/temp/test_adam/funs/funsearch_vm_adam/implementation/adam/MNIST/', train=False)

# train_loader = Data.DataLoader(
#     dataset=train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.  # shape (2000, 1, 28, 28), 归一化
# test_x = test_x.to(device)  # 移动到设备
# test_y = test_data.targets[:2000].to(device)

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=16,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         output = self.out(x)
#         return output

# cnn = CNN().to(device)
# optimizer = MyOptimizer(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()
# length = 0
# target_accuracy = 0.95
# for epoch in range(EPOCH):
#     for step, (batch_x, batch_y) in enumerate(train_loader):
#         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#         output = cnn(batch_x)
#         loss = loss_func(output, batch_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         length += 1
#         if step % 10 == 0:
#             test_output = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
#             accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
#             if accuracy >= target_accuracy:
#                 break
# print(-length)

# 下载mnist手写数据集
from torch.optim import Adam
from deepobs import pytorch as pt
from deepobs import config as global_config
from deepobs.abstract_runner.abstract_runner import Runner
from deepobs import pytorch as pt
import deepobs 
import numpy as np
# import argparse
# parser = argparse.ArgumentParser(description='adam')
# parser.add_argument('--file_name', type = str)
# args = parser.parse_args()
# file_name = args.file_name
# import pdb;pdb.set_trace()
import torch
import torch.nn as nn
import torchvision 
import torch.utils.data as Data
import time
import os
from torch.optim import Optimizer

from deepobs import config as global_config
from deepobs.abstract_runner.abstract_runner import Runner
from deepobs.pytorch.runners.runner import PTRunner
start_time = time.time()
# 超参数
EPOCH = 1       # 前向后向传播迭代次数
LR = 0.001      # 学习率 learning rate 
BATCH_SIZE = 256 # 批量训练时候一次送入数据的size
DOWNLOAD_MNIST = True 
class MyOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, file_name=None):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(MyOptimizer, self).__init__(params, defaults)
        self.file_name = os.getenv('adam_file_name')
        
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                import dill
                with open(self.file_name, 'r') as file:
                    read_function_code = file.read()
                local_vars = {}
                exec(read_function_code, globals(), local_vars)
                adam_update = local_vars['adam_update']
                adam_update([p.data], [grad], [exp_avg], [exp_avg_sq], group['lr'], beta1, beta2, group['eps'], state['step'])
# def adam_update(parameters, grads, exp_avgs, exp_avg_sqs, lr, beta1, beta2, eps, step):
#     """
#     Update parameters using the Adam optimization algorithm.

#     Parameters:
#     parameters (iterable of tensors): The parameters to be updated.
#     grads (iterable of tensors): The gradients of the parameters.
#     exp_avgs (iterable of tensors): Exponential moving averages of gradients.
#     exp_avg_sqs (iterable of tensors): Exponential moving averages of squared gradients.
#     lr (float): Learning rate.
#     beta1 (float): Exponential decay rate for the first moment estimates.
#     beta2 (float): Exponential decay rate for the second moment estimates.
#     eps (float): Small constant to prevent division by zero.
#     step (int): Current optimization step.

#     Returns:
#     None
#     """
#     """Improved version of `adam_update_v0`."""

#     # Calculate the bias corrections for the moving averages
#     for param, grad, exp_avg, exp_avg_sq in zip(parameters, grads, exp_avgs, exp_avg_sqs):
#         if grad is None:
#             continue

#         exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#         exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#         bias_correction1 = 1 - beta1 ** step
#         bias_correction2 = 1 - beta2 ** step

#         step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

#         param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
class StandardRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):
        super(StandardRunner, self).__init__(optimizer_class, hyperparameter_names)

    def training(
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
    ):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###
            # def get_acc(tproblem):
                
            #     tproblem.test_init_op()
            #     msg = "TEST:"
            #     loss = 0.0
            #     accuracy = 0.0
            #     batchCount = 0.0
            #     while True:
            #         try:
            #             batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy()
            #             batchCount += 1.0
            #             loss += batch_loss.item()
            #             accuracy += batch_accuracy
            #         except StopIteration:
            #             break

            #     loss /= batchCount
            #     accuracy /= batchCount

                
            #     return accuracy

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()
                    batch_loss.backward()
                    opt.step()
                    
                    tproblem.test_init_op()
                    acc_ = get_acc(tproblem)
                    tproblem.train_init_op()
                    if acc_ >= 0.8:
                        break
                    # print(acc_)
                    # print(batch_count)
                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
            "global_step" : global_step
        }

        return output

# 指定要删除的目录路径
directory_path = '/mnt/data/wjh/temp/test_adam/funs/funsearch_vm_adam/implementation/adam/results/fmnist_mlp/MyOptimizer/num_epochs__1__batch_size__128__betas__(0.9, 0.999)__eps__1.e-08__lr__1.e-02'
import shutil
import os

# 检查目录是否存在
if os.path.exists(directory_path):
    # 删除目录及其所有内容
    shutil.rmtree(directory_path)
    print(f"目录 {directory_path} 已被删除。")
else:
    print(f"目录 {directory_path} 不存在。")
optimizer_class = MyOptimizer  
hyperparams = {
    "lr": {"type": float, "default": 0.001},
    "betas": {"type": tuple, "default": (0.9, 0.999)},
    "eps": {"type": float, "default": 1e-8}

}

# runner = StandardRunner(optimizer_class, hyperparams) pt.runners.StandardRunner
runner = StandardRunner(optimizer_class, hyperparams)
output = runner.run(testproblem='fmnist_mlp', hyperparams={'lr': 1e-2}, num_epochs=10,print_train_iter=False)



print("---------------------------")
# analyzer = deepobs.analyzer.analyze_utils.Analyzer("results")
# deepobs.analyzer.analyze.get_best_run(analyzer)
# print(max(output["test_accuracies"]))
print(-output["global_step"])
end = time.time()
# print(str(max(output["test_accuracies"]))[:4])
# print(end-start_time)
# 结果将自动打印
