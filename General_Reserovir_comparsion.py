import torch.nn
from utils import *
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deal import load_config
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config_path = 'res_config.yaml'
config = load_config('res_config.yaml')
dataset_index = '4'
dataset_config = config[dataset_index]
for key, value in dataset_config.items():
    print(f"{key}: {value}")
name = dataset_config['name']
input_size = dataset_config['input_size']
data_split = dataset_config['data_split']
split_ratio = dataset_config['split_ratio']
print_period = dataset_config['print_period']
r_state = dataset_config['r_state']
k_value = dataset_config['k_value']
no_bias = True
r_state_only = True
if r_state_only:
    r_state = input_size + r_state
    r_output = r_state
else:
    r_output = input_size + r_state
layer1_neuron_number = dataset_config['layer1']
layer2_neuron_number = dataset_config['layer2']
layer3_neuron_number = dataset_config['layer3']
window_length = dataset_config['window_length']
batch_size = dataset_config['batch_size']
whether_mse = True
epochs = 200
is_decay_constant = True
is_early_stop = True
back_ratio = 0.1
if data_split:
    train_data_path = '/home/zzhan281/dataset/{}/dealed_data/train_data.npz'.format(name)
    test_data_path = '/home/zzhan281/dataset/{}/dealed_data/test_data.npz'.format(name)
else:
    all_data_path = dataset_config['all_data_path']
image_direct = '/home/zzhan281/dataset/{}/result/eprop/'.format(name)
point_update_period = 3
point_num = dataset_config['point_num']
voltage_lambda = 0.5
correct_ratio = 0.9
alpha = 0.9
beta = 0.9
max_gradient_record = []
min_gradient_record = []


def train(Res, snn1, snn2, snn3, train_dataloader, criterion, optimizer, device, epoch, point_array):
    print('update point:', end=' ')
    print(point_array)
    snn1.train()
    snn2.train()
    snn3.train()
    Res.to(device)
    snn1.to(device)
    snn2.to(device)
    snn3.to(device)
    s_start = time.time()
    mid_loss = 0
    samples = 0
    total_samples = 0
    temp_record = torch.zeros([window_length]).to(device)
    temp_spike_count = 0
    N = batch_size * layer3_neuron_number
    for i_batch, sample_batched in enumerate(train_dataloader):
        # if i_batch >= 320:
        #     break
        l1_alpha_grad = 0
        l1_beta_grad = 0
        l2_alpha_grad = 0
        l2_beta_grad = 0
        l3_alpha_grad = 0
        l3_beta_grad = 0
        early_stop = 0
        x_train = sample_batched[0].to(device)
        cross_target = sample_batched[1].to(device)
        mse_target = sample_batched[2].to(device)
        # print(mse_target)
        samples += x_train.shape[0]
        total_samples += x_train.shape[0]
        l1_states = snn1.create_init_states(device)
        l2_states = snn2.create_init_states(device)
        l3_states = snn3.create_init_states(device)
        res_states = Res.create_init_states(device)
        out_spike_count = torch.zeros([batch_size, layer3_neuron_number], device=device)
        l1_elig = torch.zeros(size=(batch_size, layer1_neuron_number, r_output), device=device)
        l2_elig = torch.zeros(size=(batch_size, layer2_neuron_number, layer1_neuron_number), device=device)
        l3_elig = torch.zeros(size=(batch_size, layer3_neuron_number, layer2_neuron_number), device=device)
        l1_lunar_epsilon = torch.zeros(size=(batch_size, layer1_neuron_number), device=device)  # 32x100
        l2_lunar_epsilon = torch.zeros(size=(batch_size, layer2_neuron_number), device=device)  # 32x100
        l3_lunar_epsilon = torch.zeros(size=(batch_size, layer3_neuron_number), device=device)  # 32x100
        for i in range(window_length):
            l1_psp = l1_states["psp"]  # F[t-1] in layer1 32x784
            l2_psp = l2_states["psp"]  # F[t-1] in layer2 32x100
            l3_psp = l3_states["psp"]  # F[t-1] in layer3 32x784
            l1_states['psp'].detach_()
            l2_states['psp'].detach_()
            l3_states['psp'].detach_()
            l1_states['v'].detach_()
            l2_states['v'].detach_()
            l3_states['v'].detach_()
            l1_states['output'].detach_()
            l2_states['output'].detach_()
            l3_states['output'].detach_()
            reservoir_output, res_states = Res(x_train[:, i, :], res_states)
            spike_l1, l1_states = snn1(reservoir_output, l1_states, is_training=True)
            spike_l2, l2_states = snn2(spike_l1, l2_states, is_training=True)
            spike_l3, l3_states = snn3(spike_l2, l3_states, is_training=True)
            optimizer.zero_grad()
            if whether_mse:
                loss = criterion(spike_l3, mse_target)
            else:
                loss = criterion(spike_l3, cross_target.long())
            loss.backward()
            temp_spike_count += torch.sum(spike_l1).item()
            with torch.no_grad():
                ex_l1_lunar_epsilon = l1_lunar_epsilon
                ex_l2_lunar_epsilon = l2_lunar_epsilon
                ex_l3_lunar_epsilon = l3_lunar_epsilon
                l1_lunar_epsilon = dO_dV_E(l1_states['v'], l1_states['sigma'])
                l2_lunar_epsilon = dO_dV_E(l2_states['v'], l2_states['sigma'])
                l3_lunar_epsilon = dO_dV_E(l3_states['v'], l3_states['sigma'])
                dE_dO3 = 2 * (spike_l3 - mse_target) / N
                dV3 = dE_dO3 * l3_lunar_epsilon  # Batch_size x layer3_neurons
                dF3 = dV3.matmul(snn3.weight.weight)
                dV2 = dF3 * snn3.beta.data * l2_lunar_epsilon
                dF2 = dV2.matmul(snn2.weight.weight)
                dV1 = dF2 * snn2.beta.data * l1_lunar_epsilon
                if is_decay_constant:
                    dF1 = dV1.matmul(snn1.weight.weight)
                    d_alpha3 = torch.sum(dF3 * l3_psp)
                    d_alpha2 = torch.sum(dF2 * l2_psp)
                    d_alpha1 = torch.sum(dF1 * l1_psp)
                    d_beta3 = torch.sum(dF3 * spike_l2)
                    d_beta2 = torch.sum(dF2 * spike_l1)
                    d_beta1 = torch.sum(dF1 * reservoir_output)
                l1_dw, l1_elig = dW_update(l1_states, ex_l1_lunar_epsilon, l1_states['lambda'], l1_elig, l1_psp, dV1)
                l2_dw, l2_elig = dW_update(l2_states, ex_l2_lunar_epsilon, l2_states['lambda'], l2_elig, l2_psp, dV2)
                l3_dw, l3_elig = dW_update(l3_states, ex_l3_lunar_epsilon, l3_states['lambda'], l3_elig, l3_psp, dV3)
                snn1.weight.weight.grad = l1_dw
                snn2.weight.weight.grad = l2_dw
                snn3.weight.weight.grad = l3_dw
                if (epoch + 1) % point_update_period == 0:
                    step_grad = torch.sum(torch.abs(l1_dw)) + torch.sum(torch.abs(l2_dw)) + torch.sum(torch.abs(l3_dw))
                    temp_record[i] += step_grad
                if is_decay_constant:
                    decay = 1
                    # if (epoch + 1) > point_num * point_update_period:
                    #     decay = decay / (epoch - (point_num * point_update_period) / 2)
                    l1_alpha_grad += (d_alpha1 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l1_beta_grad += (d_beta1 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l2_alpha_grad += (d_alpha2 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l2_beta_grad += (d_beta2 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l3_alpha_grad += (d_alpha3 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l3_beta_grad += (d_beta3 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    snn1.alpha.grad = l1_alpha_grad * decay
                    snn1.beta.grad = l1_beta_grad * decay
                    snn2.alpha.grad = l2_alpha_grad * decay
                    snn2.beta.grad = l2_beta_grad * decay
                    snn3.alpha.grad = l3_alpha_grad * decay
                    snn3.beta.grad = l3_beta_grad * decay
            out_spike_count += spike_l3
            if i in point_array:
                optimizer.step()
                snn1.alpha.data.clamp_(0, 1)
                snn2.alpha.data.clamp_(0, 1)
                snn3.alpha.data.clamp_(0, 1)
                if is_early_stop:
                    if epoch >= (point_num * point_update_period) and confidence(out_spike_count, cross_target,
                                                                                  batch_size, correct_ratio):
                        early_stop += 1
                        if early_stop >= (point_num / 2):
                            continue
            mid_loss += loss.item()
        if (i_batch+1) % print_period == 0:
            s_end = time.time()
            mid_loss /= print_period
            l1_b = snn1.beta.data
            l2_b = snn2.beta.data
            l3_b = snn3.beta.data
            l1_a = snn1.alpha.data
            l2_a = snn2.alpha.data
            l3_a = snn3.alpha.data
            l1_v = snn1.voltage_decay.data
            l2_v = snn2.voltage_decay.data
            l3_v = snn3.voltage_decay.data
            times = s_end - s_start
            print(
                "Training epoch{}/{}: [{}/{}]\tLoss: {} \t beta:{:.2f}/{:.2f}/{:.2f} \t alpha:{:.2f}/{:.2f}/{:.2f} \tlambda{:.2f}/{:.2f}/{:.2f} \tTime consumption: {:.8}s ".format(
                    epoch, epochs, i_batch + 1, len(train_dataloader), mid_loss, l1_b, l2_b, l3_b,
                    l1_a, l2_a, l3_a, l1_v, l2_v, l3_v, times))
            mid_loss = 0
            s_start = time.time()
    avg_spike_count = temp_spike_count / total_samples
    if (epoch + 1) % point_update_period == 0:
        _, idx = torch.sort(temp_record, descending=True)
        point_array = idx[0:point_num].cpu().detach().numpy()
        max_index = point_array[0]
        last_record = point_array[0: point_num - 1]
        if max_index == window_length - 1:
            max_index = point_array[1]
        return last_record, max_index, avg_spike_count
    else:
        return 0, 0, avg_spike_count


def test(Res, snn1, snn2, snn3, test_dataloader, criterion, device, epoch):
    Res.eval()
    snn1.eval()
    snn2.eval()
    snn3.eval()
    test_loss = 0
    acc = 0
    s = time.time()
    samples = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dataloader):
            # if i_batch >= 20:
            #     break
            x_test = sample_batched[0].to(device)
            cross_target = sample_batched[1].to(device)
            mse_target = sample_batched[2].to(device)
            l1_states = snn1.create_init_states(device)
            l2_states = snn2.create_init_states(device)
            l3_states = snn3.create_init_states(device)
            res_states = Res.create_init_states(device)
            out_spike_count = torch.zeros([batch_size, layer3_neuron_number], device=device)
            for i in range(window_length):
                reservoir_output, res_states = Res(x_test[:, i, :], res_states)
                spike_l1, l1_states = snn1(reservoir_output, l1_states, is_training=False)
                spike_l2, l2_states = snn2(spike_l1, l2_states, is_training=False)
                spike_l3, l3_states = snn3(spike_l2, l3_states, is_training=False)
                out_spike_count += spike_l3
            if whether_mse:
                loss = criterion(out_spike_count, mse_target * window_length)
            else:
                loss = criterion(out_spike_count, cross_target.long())
            test_loss += loss
            max_index = torch.argmax(out_spike_count, dim=1, keepdim=False)
            correct_index = max_index == cross_target
            acc += sum(correct_index)
            samples += x_test.shape[0]
        test_loss = test_loss.item()
        test_loss /= len(test_dataloader)
        avg_acc = acc.item() / samples
        e = time.time()
        print('Average test loss {:.6f}\tAverage accuracy {:.6f}\ttime consumption for test function: {:.8}s'.format(
            test_loss, avg_acc, e - s))
        return avg_acc


def full_training(train_loader, test_loader, device, path, time_index):
    Res = R_Sp_Link_SNN(input_size, r_output, r_state, batch_size, voltage_lambda, no_bias=no_bias, k_value=k_value,
                        r_state_only=r_state_only)
    snn1 = ESNN(r_output, layer1_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    snn2 = ESNN(layer1_neuron_number, layer2_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    snn3 = ESNN(layer2_neuron_number, layer3_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    params = list(snn1.parameters()) + list(snn2.parameters()) + list(snn3.parameters())
    for p in Res.parameters():
        p.requires_grad = False
    optimizer = torch.optim.RMSprop(params, lr=0.0001)
    criterion = torch.nn.MSELoss()
    epoch_list = np.arange(epochs).tolist()
    point_array = np.ones(point_num) * (window_length - 1)
    last_record = np.ones(point_num - 1) * (window_length - 1)
    point_array[0:point_num - 1] = last_record

    best_acc = 0
    acc_record = []
    temp_index = 0
    temp_array = (torch.ones(point_num) * (window_length - 1)).to(device)
    snn1_dict = {}
    snn2_dict = {}
    snn3_dict = {}
    epoch_list = np.arange(epochs).tolist()
    for epoch in range(epochs):
        last_record, max_index, avg_spike_count = train(Res, snn1, snn2, snn3, train_loader, criterion, optimizer,
                                                        device, epoch,
                                                        point_array)
        if (epoch + 1) % point_update_period == 0 and temp_index < point_num - 1:
            point_array[0:point_num - 1] = last_record
            temp_array[temp_index] = max_index
            temp_index += 1
            if temp_index == point_num - 1:
                point_array = temp_array
        acc = test(Res, snn1, snn2, snn3, test_loader, criterion, device, epoch)
        acc_record.append(acc)
        if acc > best_acc:
            best_acc = acc
            snn1_dict = snn1.state_dict()
            snn2_dict = snn2.state_dict()
            snn3_dict = snn3.state_dict()
        print('best acc:{}'.format(best_acc))
    snn_name1 = 'S{}_snn1_{}_{}.pkl'.format(time_index, input_size, layer1_neuron_number)
    snn_name2 = 'S{}_snn2_{}_{}.pkl'.format(time_index, layer1_neuron_number, layer2_neuron_number)
    snn_name3 = 'S{}_snn3_{}_{}.pkl'.format(time_index, layer2_neuron_number, layer3_neuron_number)
    checkpoint1 = os.path.join(path, snn_name1)
    checkpoint2 = os.path.join(path, snn_name2)
    checkpoint3 = os.path.join(path, snn_name3)
    torch.save(snn1_dict, checkpoint1)
    torch.save(snn2_dict, checkpoint2)
    torch.save(snn3_dict, checkpoint3)
    print('Successfull save model {} to {}'.format(snn_name1, checkpoint1))
    print('Successfull save model {} to {}'.format(snn_name2, checkpoint2))
    print('Successfull save model {} to {}'.format(snn_name3, checkpoint3))
    image_name = 'S{}_{}_Res_Schedule_Eprop_{}_{}_class_{:.4f}.jpg'.format(time_index, name, window_length, layer3_neuron_number, best_acc)
    image_path = os.path.join(path, image_name)
    plt.plot(epoch_list, acc_record, color='red', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('{} Res Schedule Eprop best accuracy: {}'.format(name, best_acc), fontsize='large')
    plt.savefig(image_path)
    plt.show()
    plt.close()
    return best_acc, Res.state_dict(), [snn1_dict, snn2_dict, snn3_dict]

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if data_split:
        train_data = load_dataset(train_data_path, layer3_neuron_number)
        test_data = load_dataset(test_data_path, layer3_neuron_number)
    else:
        all_data = load_dataset(all_data_path, layer3_neuron_number)
        train_size = int(split_ratio * len(all_data))
        test_size = len(all_data) - train_size
        train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    acc_list = []
    Res_list = []
    SNN_list = []
    times = 3
    Encoders = []
    Decoders = []
    i = 0
    while 1:
        filename = 'Res_comparison_{}_{}'.format(name, i)
        path = os.path.join(image_direct, filename)
        if not os.path.exists(path):
            os.makedirs(path)
            break
        i += 1
    for i in range(times):
        acc, Res_dict, SNN_dicts = full_training(train_loader, test_loader, device, path, i)
        acc_list.append(acc)
        Res_list.append(Res_dict)
        SNN_list.append(SNN_dicts)
        Encoders.append('Reservoir {}'.format(i))
        Decoders.append('Decoder {}'.format(i))

    Res = R_Sp_Link_SNN(input_size, r_output, r_state, batch_size, voltage_lambda, p=p_p, p_self=p_self,
                        no_cycle=False, no_bias=no_bias, k_value=k_value, r_state_only=r_state_only)
    snn1 = ESNN(r_output, layer1_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    snn2 = ESNN(layer1_neuron_number, layer2_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    snn3 = ESNN(layer2_neuron_number, layer3_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                is_voltage_decay_constant=False, alpha=alpha, beta=beta)
    criterion = torch.nn.MSELoss()
    accuracy_matrix = pd.DataFrame(index=Encoders, columns=Decoders)
    for e in range(times):
        for d  in range(times):
            encoder = Encoders[e]
            decoder = Decoders[d]
            if e == d:
                accuracy_matrix.loc[encoder, decoder] = acc_list[e]
            else:
                res_dict = Res_list[e]
                SNN_dicts = SNN_list[d]
                snn1_dict = SNN_dicts[0]
                snn2_dict = SNN_dicts[1]
                snn3_dict = SNN_dicts[2]
                Res.load_state_dict(res_dict)
                snn1.load_state_dict(snn1_dict)
                snn2.load_state_dict(snn2_dict)
                snn3.load_state_dict(snn3_dict)
                Res.to(device)
                snn1.to(device)
                snn2.to(device)
                snn3.to(device)
                acc = test(Res, snn1, snn2, snn3, test_loader, criterion, device, epoch=0)
                accuracy_matrix.loc[encoder, decoder] = acc
    accuracy_matrix = accuracy_matrix.astype(float)
    image_name = '{}_Res_heatmap.jpg'.format(name)
    image_path = os.path.join(path, image_name)
    plot_accuracy_matrix(accuracy_matrix, image_path)

if __name__ == '__main__':
    main()