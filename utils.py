import torch
import numpy as np
import time
import seaborn as sns
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import psutil
import cv2
import networkx as nx
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn


class threshold(torch.autograd.Function):
    """
    heaviside step threshold function
    """

    @staticmethod
    def forward(ctx, input, sigma):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        ctx.sigma = sigma

        output = input.clone()

        # gpu: 0.182
        # cpu: 0.143
        # output[output < 1] = 0
        # output[output >= 1] = 1

        # gpu: 0.157s
        # cpu: 0.137
        output = torch.max(torch.tensor(0.0, device=output.device), torch.sign(output - 1.0))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # grad_output is dE/dO
        # a = time.time()
        input, = ctx.saved_tensors
        sigma = ctx.sigma

        # start = time.time()
        exponent = -torch.pow((1 - input), 2) / (2.0 * sigma ** 2)
        exp = torch.exp(exponent)
        erfc_grad = exp / (2.5066282746310007 * sigma)  # gradient for dU/dv
        grad = erfc_grad * grad_output

        return grad, None


class load_dataset(Dataset):
    def __init__(self, data_path, classes):
        all_data = np.load(data_path)
        data = all_data['data']
        print(data.shape)
        isnan = np.isnan(data)
        if (True in isnan):
            data = np.nan_to_num(data)
            print('nan value exist')
        self.data = data  # num x 1197 x 6
        self.label = all_data['label']
        self.classes = classes

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.tensor(self.data[idx, :, :], dtype=torch.float32)
        cross_label = torch.tensor(self.label[idx], dtype=torch.int64)
        mse_label = torch.zeros(self.classes)
        mse_label[cross_label] = 1
        return data, cross_label, mse_label

# Frequency testing dataloader
class MTSCDataset(Dataset):
    def __init__(self, data_path, classes, original_window_length, rated_window_length):
        all_data = np.load(data_path)
        data = all_data['data']
        isnan = np.isnan(data)
        if (True in isnan):
            data = np.nan_to_num(data)
            print('nan value exist')
        self.data = data
        self.label = all_data['label']
        self.classes = classes
        self.rated_window_length = rated_window_length
        self.original_window_length = original_window_length

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx, :, :]

        if self.rated_window_length != self.original_window_length:
            rated_data = np.zeros((self.rated_window_length, data.shape[1]))
            import math
            for i in range(self.rated_window_length):
                location = i / self.rated_window_length * self.original_window_length
                lower = math.floor(location)
                upper = min(math.ceil(location), self.original_window_length-1)
                left_distance = location - lower
                right_distance = 1 - left_distance
                #print(i,location,lower,upper,left_distance,right_distance)
                rated_data[i,:] = data[lower,:]*right_distance + data[upper,:]*left_distance
            data = torch.tensor(rated_data, dtype=torch.float32)
        cross_label = torch.tensor(self.label[idx], dtype=torch.int64)
        mse_label = torch.zeros(self.classes)
        mse_label[cross_label] = 1
        return data, cross_label, mse_label


def dW_update(cur_layer_state, l_lunate_epsilon,
               voltage_lambda, elig_trace, psp, dV):
    # to simplify the computation, I directly use dE/dV to replace dE/dO*dO/dV
    # l_lunar_epsilon is ϵ[t-1] in layer l

    # calculate learning signal
    plus1_u = dV
    # plus1_u = torch.where(torch.isnan(plus1_u), torch.full_like(plus1_u, 0), plus1_u)
    # print(plus1_u)
    plus1_u_extend = plus1_u.unsqueeze(2)

    # calculate eligibility trace, l_lunar_epsilon_E 32x100
    sigma = cur_layer_state["sigma"]
    l_lunate_epsilon_ext = l_lunate_epsilon.unsqueeze(2)
    psp_extend = psp.unsqueeze(1)
    # voltage_lambda = voltage_lambda.unsqueeze(2)
    elig_trace_update = (voltage_lambda - sigma * l_lunate_epsilon_ext) * elig_trace + psp_extend

    batch_dw = (plus1_u_extend * elig_trace_update)  # x
    batch_size = batch_dw.shape[0]
    dw = batch_dw.sum(dim=0)
    dw = dw / batch_size
    # dw = dw
    # if there is no detach function. then dw will be a tensor require grad. so it cannot be value to a weight's grad

    return dw.detach(), elig_trace_update.detach()


def confidence(output_spike, cross_target, batch_size, probability_threshold, sample_ratio=0.9):
    temp = torch.sum(output_spike, dim=1, keepdim=True)
    output_spike_score = output_spike / temp
    (probability, predicted_class) = output_spike_score.max(dim=1)
    difference = predicted_class - cross_target
    index = difference == 0
    correct_samples = torch.sum(index)  # how many samples predict correct
    correct_ratio = correct_samples / batch_size
    correct_probability = probability[index]
    # how many samples have very high predict probability in correct samples
    correct_ratio1 = torch.sum(correct_probability >= probability_threshold) / correct_samples
    if correct_ratio > sample_ratio and correct_ratio1 > sample_ratio:
        return True  # prediction acceptable, stop training
    else:
        return False  # prediction unacceptable, keep training


class R_Sp_Link_SNN(torch.nn.Module):
    def __init__(self, input_size, output_size, r_states, batch_size=4, voltage_decay=0.5, no_bias=False, k_value=0.9, r_state_only=False):
        super().__init__()

        # input_size is K, r_states is N, output_size is L
        self.w_in_t = torch.nn.Parameter(torch.rand(size=(r_states, input_size)).t())  # K x N
        weighted_adj_matrix = torch.rand(size=(r_states, r_states))
        self.k = torch.nn.Parameter(torch.tensor(k_value))
        self.w_t = torch.nn.Parameter(torch.nn.Parameter(torch.rand(size=(r_states, r_states))))  # N x N
        self.bias = torch.nn.Parameter(torch.rand(size=(1, r_states)))  # 1 x N, broadcast
        torch.nn.init.xavier_uniform_(self.w_in_t.data)
        torch.nn.init.xavier_uniform_(self.w_t.data)
        self.voltage_decay = torch.nn.Parameter(torch.tensor(voltage_decay))
        self.sigma = torch.nn.Parameter(torch.tensor(0.9))
        self.input_size = input_size
        self.r_states = r_states
        self.output_size = output_size
        self.batch_size = batch_size
        self.tanh = torch.nn.Tanh()
        self.no_bias = no_bias
        self.r_states_only = r_state_only

    def forward(self, u_n, x_state):
        prev_x_n = x_state['state']  # batch x N
        # prev_y_n = x_state['output']  # batch x L, should be the output signal of the whole network.
        prev_v = x_state['v']
        prev_spike = x_state["output"]
        if self.no_bias:
            x_n = (1 - self.k) * torch.matmul(prev_x_n, self.w_t) + self.k * torch.matmul(u_n, self.w_in_t)
        else:
            # print('call this')
            x_n = (1 - self.k) * torch.matmul(prev_x_n, self.w_t) + self.k * torch.matmul(u_n, self.w_in_t) + self.bias
        if not self.r_states_only:
            z_n = torch.cat((u_n, x_n), dim=1)  # batch x ( N + K)
            input_channels = u_n.shape[-1]
            prev_spike[:, 0:input_channels] = 0
        else:
            z_n = x_n
            if prev_spike.shape[-1] != self.r_states:
                input_channels = u_n.shape[-1]
                prev_spike = prev_spike[:, input_channels:]
        current_v = z_n - self.sigma * prev_spike
        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)  # batch x ( N + K)
        new_state = {"state": x_n.detach_(), "output": spike.detach_(), "v": current_v.detach_()}

        return spike.detach_(), new_state

    def create_init_states(self, device, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        x_n = torch.zeros([batch_size, self.r_states], device=device)
        voltage = torch.zeros([batch_size, self.r_states + self.input_size], device=device)
        spike = torch.zeros([batch_size, self.r_states + self.input_size], device=device)
        states = {"state": x_n, "output": spike, "v": voltage}

        return states

    def save_data(self, path):
        w_in_t = self.w_in_t.cpu().detach().numpy()
        w_t = self.w_t.cpu().detach().numpy()
        name = 'w_in.txt'
        name1 = 'w.txt'
        name2 = 'bias.txt'
        array_path = os.path.join(path, name)
        array_path1 = os.path.join(path, name1)
        array_path2 = os.path.join(path, name2)
        np.savetxt(array_path, w_in_t, fmt='%f', delimiter=',')
        np.savetxt(array_path1, w_t, fmt='%f', delimiter=',')
        if not self.no_bias:
            bias = self.bias.cpu().detach().numpy()
            np.savetxt(array_path2, bias, fmt='%f', delimiter=',')
        print('matrix saved')

    def load_data(self, w_path, win_path, bias_path=None):
        load_w = np.loadtxt(w_path, delimiter=',')
        load_win = np.loadtxt(win_path, delimiter=',')
        self.w_t = torch.nn.Parameter(torch.tensor(load_w, dtype=torch.float32))
        self.w_in_t = torch.nn.Parameter(torch.tensor(load_win, dtype=torch.float32))
        if bias_path:
            load_bias = np.loadtxt(bias_path)
            self.bias = torch.tensor(load_bias)


class ESNN(torch.nn.Module):
    def __init__(self, input_size, neuron_number, batch_size=10, voltage_lambda=0.2, is_decay_constant=False, alpha=0.9,
                 beta=0.9, is_voltage_decay_constant=False):
        super().__init__()
        self.input_size = input_size
        self.neuron_number = neuron_number
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.alpha.requires_grad = is_decay_constant
        self.beta.requires_grad = is_decay_constant
        self.batch_size = batch_size
        self.weight = torch.nn.Linear(input_size, neuron_number, bias=False)

        # sigma is Vth, using to compute gradient and current voltage
        self.sigma = torch.nn.Parameter(torch.tensor(0.5))
        self.sigma.requires_grad = False

        # voltage_decay is λ(lambda)
        # self.voltage_decay = np.full(neuron_number, voltage_lambda, dtype=np.float32)
        self.voltage_decay = voltage_lambda
        self.voltage_decay = torch.nn.Parameter(torch.tensor(self.voltage_decay))
        self.voltage_decay.requires_grad = is_voltage_decay_constant

    def forward(self, current_spike, state_dict, is_training=False):
        # actually input spike is current_spike, not psp spike. psp spike initial value is zero
        prev_v = state_dict["v"]
        prev_spike = state_dict["output"]
        prev_psp = state_dict["psp"]
        current_psp = self.alpha * prev_psp + self.beta * current_spike

        current_v = self.voltage_decay * prev_v + self.weight(current_psp) - self.sigma * prev_spike

        threshold_function = threshold.apply
        spike = threshold_function(current_v, self.sigma)  # output spike
        if is_training:
            current_v.retain_grad()
            spike.retain_grad()

        new_state = {"v": current_v, "psp": current_psp, "output": spike, "beta": self.beta, "alpha": self.alpha,
                     "weight": self.weight,
                     "sigma": self.sigma, "lambda": self.voltage_decay}
        return spike, new_state

    def create_init_states(self, device, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        voltage = torch.zeros([batch_size, self.neuron_number], device=device)
        psp = torch.zeros([batch_size, self.input_size], device=device)
        spikes = torch.zeros([batch_size, self.neuron_number], device=device)

        states = {"v": voltage, "psp": psp, "output": spikes}
        return states


class LSTM_Baseline(torch.nn.Module):
    def __init__(self, INPUT_SIZE, lstm_size, NUM_OF_CLASS):
        super(LSTM_Baseline, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=INPUT_SIZE, hidden_size=lstm_size, num_layers=1, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(lstm_size, lstm_size)
        self.fc2 = torch.nn.Linear(lstm_size, NUM_OF_CLASS)

    def forward(self, x):
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x)  # lstm with input, hidden, and internal cell state
        out = output
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sum(out,1)
        return out


def dO_dV_E(voltage, sigma):
    # dO_dV: 32x100
    exponent = -torch.pow((1 - voltage), 2) / (2.0 * sigma ** 2)
    exp = torch.exp(exponent)
    erfc_grad = exp / (2.5066282746310007 * sigma)  # gradient for dU/dv
    return erfc_grad.detach()


# Heaviside Activation
class HeavisideActivation(nn.Module):
    def forward(self, x):
        return torch.heaviside(x, torch.tensor(0.0))


# 1. 随机稀疏连接 Reservoir
class StepByStepRandomSparseReservoir(nn.Module):
    def __init__(self, input_size, output_size, sparsity=0.1):
        super(StepByStepRandomSparseReservoir, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sparsity = sparsity
        self.reservoir_weights = nn.Parameter(
            torch.rand(output_size, output_size) * (torch.rand(output_size, output_size) < sparsity),
            requires_grad=False
        )
        self.input_weights = nn.Parameter(torch.rand(output_size, input_size), requires_grad=False)
        self.heaviside = HeavisideActivation()
        self.reservoir_state = None  # 延迟初始化

    def forward(self, x):
        device = next(self.parameters()).device  # 确保设备一致
        if self.reservoir_state is None or self.reservoir_state.size(0) != x.size(0):
            self.reset_state(x.size(0), device)

        x = x.to(device)
        self.input_weights = self.input_weights.to(device)
        self.reservoir_weights = self.reservoir_weights.to(device)
        self.reservoir_state = self.reservoir_state.to(device)

        self.reservoir_state = torch.tanh(
            torch.matmul(x, self.input_weights.T) + torch.matmul(self.reservoir_state, self.reservoir_weights)
        )
        spike_output = self.heaviside(self.reservoir_state)
        return spike_output

    def reset_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        self.reservoir_state = torch.zeros(batch_size, self.output_size, device=device)


# 2. 循环结构 Reservoir
class StepByStepCyclicReservoir(nn.Module):
    def __init__(self, input_size, output_size):
        super(StepByStepCyclicReservoir, self).__init__()
        self.input_weights = nn.Parameter(torch.rand(output_size, input_size), requires_grad=False)

        # 创建循环连接矩阵
        cycle_matrix = torch.zeros(output_size, output_size)  # 初始化为全零矩阵
        cycle_matrix += torch.diag(torch.ones(output_size - 1), diagonal=1)  # 上一对角线
        cycle_matrix += torch.diag(torch.ones(1), diagonal=-output_size + 1)  # 环绕下对角线

        self.cycle_connections = nn.Parameter(cycle_matrix, requires_grad=False)
        self.heaviside = HeavisideActivation()
        self.reservoir_state = None

    def reset_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        self.reservoir_state = torch.zeros(batch_size, self.input_weights.size(0), device=device)

    def forward(self, x):
        if self.reservoir_state is None:
            self.reset_state(x.size(0), x.device)
        self.reservoir_state = torch.tanh(
            torch.matmul(x, self.input_weights.T) + torch.matmul(self.reservoir_state, self.cycle_connections)
        )
        spike_output = self.heaviside(self.reservoir_state)
        return spike_output


# 3. 层次结构 Reservoir
class StepByStepHierarchicalReservoir(nn.Module):
    def __init__(self, input_size, output_size, num_layers=3, scale_factor=10.0, bias=0.1):
        super(StepByStepHierarchicalReservoir, self).__init__()
        self.num_layers = num_layers
        self.scale_factor = scale_factor
        self.bias = bias

        # 输入权重矩阵
        self.input_weights = nn.Linear(input_size, output_size, bias=False)
        nn.init.uniform_(self.input_weights.weight, a=-2.0, b=2.0)

        # 每层的递归权重矩阵
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(output_size, output_size, bias=False)
            nn.init.uniform_(layer.weight, a=-2.0, b=2.0)
            layer.weight.requires_grad = False
            self.layers.append(layer)

        self.heaviside = HeavisideActivation()
        self.reservoir_states = None

    def reset_state(self, batch_size, output_size, device=None):
        device = device or next(self.parameters()).device
        # 为每一层的状态初始化独立的张量
        self.reservoir_states = [
            torch.zeros(batch_size, output_size, device=device) for _ in range(self.num_layers)
        ]

    def forward(self, x):
        x = x * self.scale_factor  # 放大信号
        device = next(self.parameters()).device
        x = x.to(device)

        # 输入信号经过输入权重映射
        input_mapped = self.input_weights(x)

        # 初始化 reservoir_states
        if self.reservoir_states is None or len(self.reservoir_states) != self.num_layers:
            self.reset_state(x.size(0), input_mapped.size(1), device)

        # 计算每一层的状态
        summed_states = torch.zeros_like(input_mapped)  # 用于累加各层状态
        for i, layer in enumerate(self.layers):
            self.reservoir_states[i] = torch.tanh(layer(self.reservoir_states[i]))
            summed_states += self.reservoir_states[i]  # 累加每层状态

        # 加和后的状态与输入信号映射结果相加，再过激活函数
        final_state = torch.tanh(input_mapped + summed_states) + self.bias

        # 输出 spike 信号
        spike_output = self.heaviside(final_state)
        return spike_output


# 4. 模块化结构 Reservoir
class StepByStepModularReservoir(nn.Module):
    def __init__(self, input_size, output_size, num_modules=5):
        super(StepByStepModularReservoir, self).__init__()
        self.reservoir_modules = nn.ModuleList()  # 使用 ModuleList 存储子模块
        self.output_size = output_size  # 目标输出尺寸
        module_output_size = output_size // num_modules  # 每个模块的输出维度
        remainder = output_size % num_modules  # 如果 output_size 不能整除 num_modules

        # 初始化模块
        for i in range(num_modules):
            # 最后一个模块加上余数 remainder 确保总维度一致
            final_output_size = module_output_size + (1 if i < remainder else 0)
            layer = nn.Linear(input_size, final_output_size, bias=False)
            nn.init.uniform_(layer.weight, a=-0.5, b=0.5)  # 初始化权重
            self.reservoir_modules.append(layer)

        for module in self.reservoir_modules:
            module.weight.requires_grad = False

        self.heaviside = HeavisideActivation()  # 使用 Heaviside 激活函数
        self.reservoir_states = None  # 存储每个模块的状态

    def reset_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        # 初始化每个模块的状态
        self.reservoir_states = [
            torch.zeros(batch_size, module.out_features, device=device)
            for module in self.reservoir_modules
        ]

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)

        # 如果状态未初始化，则重置
        if self.reservoir_states is None:
            self.reset_state(x.size(0), device)

        module_outputs = []
        for i, module in enumerate(self.reservoir_modules):
            self.reservoir_states[i] = torch.tanh(module(x))  # 使用 tanh 激活
            module_outputs.append(self.reservoir_states[i])

        # 拼接所有模块的输出
        combined_output = torch.cat(module_outputs, dim=-1)
        spike_output = self.heaviside(combined_output)  # 应用 Heaviside 函数
        return spike_output


# 5. 距离约束结构 Reservoir
class StepByStepDistanceConstrainedReservoir(nn.Module):
    def __init__(self, input_size, output_size, distance_decay=0.5):
        super(StepByStepDistanceConstrainedReservoir, self).__init__()
        self.input_weights = nn.Parameter(torch.rand(output_size, input_size), requires_grad=False)
        distances = torch.abs(torch.arange(output_size).view(-1, 1) - torch.arange(output_size).view(1, -1))
        self.reservoir_weights = nn.Parameter(
            torch.exp(-distance_decay * distances) * (torch.rand(output_size, output_size) < 0.5),
            requires_grad=False
        )
        self.heaviside = HeavisideActivation()
        self.reservoir_state = None

    def reset_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        self.reservoir_state = torch.zeros(batch_size, self.input_weights.size(0), device=device)

    def forward(self, x):
        if self.reservoir_state is None:
            self.reset_state(x.size(0), x.device)
        self.reservoir_state = torch.tanh(
            torch.matmul(x, self.input_weights.T) + torch.matmul(self.reservoir_state, self.reservoir_weights)
        )
        spike_output = self.heaviside(self.reservoir_state)
        return spike_output


def plot_accuracy_matrix(accuracy_matrix, image_path):
    """
    绘制 encoder-decoder 组合的准确度热力图，图的大小根据矩阵的大小动态调整。

    参数:
    accuracy_matrix (pd.DataFrame): 记录 encoder 和 decoder 组合的准确度矩阵，index 是 encoders，columns 是 decoders。

    返回:
    None
    """
    # 获取矩阵的大小
    num_encoders = accuracy_matrix.shape[0]
    num_decoders = accuracy_matrix.shape[1]

    # 动态调整图的大小，每个单元格设定为一个合适的尺寸
    cell_size = 1.0  # 每个单元格的大小
    fig_width = cell_size * num_decoders + 2  # 根据解码器数量调整宽度
    fig_height = cell_size * num_encoders + 2  # 根据编码器数量调整高度

    # 绘制热力图展示 encoder-decoder 组合的准确度
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(accuracy_matrix, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Accuracy of Encoder-Decoder Combinations')
    plt.xlabel('Decoders')
    plt.ylabel('Encoders')
    plt.show()
    plt.savefig(image_path)