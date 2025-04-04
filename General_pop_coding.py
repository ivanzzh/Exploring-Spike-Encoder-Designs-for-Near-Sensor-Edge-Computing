import torch.nn
from utils import *
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from deal import *


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
config_path = 'config.yaml'
config = load_config('config.yaml')
dataset_index = '4'
dataset_config = config[dataset_index]
for key, value in dataset_config.items():
    print(f"{key}: {value}")
name = dataset_config['name']
input_size = dataset_config['input_size']
data_split = dataset_config['data_split']
split_ratio = dataset_config['split_ratio']
layer1_neuron_number = dataset_config['layer1']
layer2_neuron_number = dataset_config['layer2']
layer3_neuron_number = dataset_config['layer3']
window_length = dataset_config['window_length']
batch_size = dataset_config['batch_size']
# batch_size = 2
point_num = dataset_config['point_num']
print_period = dataset_config['print_period']
N = batch_size * layer3_neuron_number
whether_mse = True
normamlized = False
epochs = 200
is_decay_constant = True
is_early_stop = True
back_ratio = 0.1
alpha = 0.9
beta = 0.9
voltage_lambda = 0.5
if data_split:
    train_data_path = '/home/zzhan281/dataset/{}/dealed_data/train_data.npz'.format(name)
    test_data_path = '/home/zzhan281/dataset/{}/dealed_data/test_data.npz'.format(name)
else:
    all_data_path = dataset_config['all_data_path']
image_direct = '/home/zzhan281/dataset/{}/result/eprop/'.format(name)
point_update_period = 3
correct_ratio = 0.9
max_gradient_record = []
min_gradient_record = []
stop_points = []



def train(snn1, snn2, snn3, train_dataloader, criterion, optimizer, device, epoch, point_array):
    print('update point:', end=' ')
    print(point_array)
    snn1.train()
    snn2.train()
    snn3.train()
    snn1.to(device)
    snn2.to(device)
    snn3.to(device)
    s_start = time.time()
    mid_loss = 0
    samples = 0
    total_samples = 0
    temp_record = torch.zeros([window_length]).to(device)
    end_point = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        # if i_batch >= 3:
        #     break
        l1_alpha_grad = 0
        l1_beta_grad = 0
        l2_alpha_grad = 0
        l2_beta_grad = 0
        l3_alpha_grad = 0
        l3_beta_grad = 0
        early_stop = 0
        sample_end = 0
        x_train = sample_batched[0].to(device)
        cross_target = sample_batched[1].to(device)
        mse_target = sample_batched[2].to(device)
        samples += x_train.shape[0]
        sample_batch = x_train.shape[0]
        total_samples += x_train.shape[0]
        l1_states = snn1.create_init_states(device, sample_batch)
        l2_states = snn2.create_init_states(device, sample_batch)
        l3_states = snn3.create_init_states(device, sample_batch)
        out_spike_count = torch.zeros([sample_batch, layer3_neuron_number], device=device)
        l1_elig = torch.zeros(size=(sample_batch, layer1_neuron_number, input_size), device=device)
        l2_elig = torch.zeros(size=(sample_batch, layer2_neuron_number, layer1_neuron_number), device=device)
        l3_elig = torch.zeros(size=(sample_batch, layer3_neuron_number, layer2_neuron_number), device=device)
        l1_lunar_epsilon = torch.zeros(size=(sample_batch, layer1_neuron_number), device=device)  # 32x100
        l2_lunar_epsilon = torch.zeros(size=(sample_batch, layer2_neuron_number), device=device)  # 32x100
        l3_lunar_epsilon = torch.zeros(size=(sample_batch, layer3_neuron_number), device=device)  # 32x100
        N = batch_size * layer3_neuron_number
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
            spike_l1, l1_states = snn1(x_train[:, i, :], l1_states, is_training=True)
            spike_l2, l2_states = snn2(spike_l1, l2_states, is_training=True)
            spike_l3, l3_states = snn3(spike_l2, l3_states, is_training=True)
            optimizer.zero_grad()
            if whether_mse:
                loss = criterion(spike_l3, mse_target)
            else:
                loss = criterion(spike_l3, cross_target.long())
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
                    d_beta1 = torch.sum(dF1 * x_train[:, i, :])
                l1_dw, l1_elig = dW_update(l1_states, ex_l1_lunar_epsilon, l1_states['lambda'], l1_elig, l1_psp, dV1)
                l2_dw, l2_elig = dW_update(l2_states, ex_l2_lunar_epsilon, l2_states['lambda'], l2_elig, l2_psp, dV2)
                l3_dw, l3_elig = dW_update(l3_states, ex_l3_lunar_epsilon, l3_states['lambda'], l3_elig, l3_psp, dV3)
                if (epoch + 1) % point_update_period == 0:
                    step_grad = torch.sum(torch.abs(l1_dw)) + torch.sum(torch.abs(l2_dw)) + torch.sum(torch.abs(l3_dw))
                    temp_record[i] += step_grad
                if is_decay_constant:
                    l1_alpha_grad += (d_alpha1 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l1_beta_grad += (d_beta1 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l2_alpha_grad += (d_alpha2 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l2_beta_grad += (d_beta2 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l3_alpha_grad += (d_alpha3 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
                    l3_beta_grad += (d_beta3 * (1 - back_ratio ** (i + 1)) / (1 - back_ratio))
            out_spike_count += spike_l3
            if i in point_array:
                snn1.weight.weight.grad = l1_dw
                snn2.weight.weight.grad = l2_dw
                snn3.weight.weight.grad = l3_dw
                snn1.alpha.grad = l1_alpha_grad
                snn1.beta.grad = l1_beta_grad
                snn2.alpha.grad = l2_alpha_grad
                snn2.beta.grad = l2_beta_grad
                snn3.alpha.grad = l3_alpha_grad
                snn3.beta.grad = l3_beta_grad
                optimizer.step()
                snn1.alpha.data.clamp_(0.1, 1)
                snn2.alpha.data.clamp_(0.1, 1)
                snn3.alpha.data.clamp_(0.1, 1)
                snn1.beta.data.clamp_(min=0.1)
                snn2.beta.data.clamp_(min=0.1)
                snn3.beta.data.clamp_(min=0.1)
                if is_early_stop:
                    if epoch >= (point_num * point_update_period) and confidence(out_spike_count, cross_target,
                                                                                  batch_size, correct_ratio):
                        early_stop += 1
                        if early_stop >= (point_num / 2):
                            sample_end = i
                            continue
            mid_loss += loss.item()
        if (i_batch + 1) % print_period == 0:
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
                "Training epoch{}/{}: [{}/{}]\tLoss: {:.3f} \t beta:{:.2f}/{:.2f}/{:.2f} \t alpha:{:.2f}/{:.2f}/{:.2f} \tlambda{:.2f}/{:.2f}/{:.2f} \tTime consumption: {:.3}s ".format(
                    epoch, epochs, i_batch + 1, len(train_dataloader), mid_loss, l1_b, l2_b, l3_b,
                    l1_a, l2_a, l3_a, l1_v, l2_v, l3_v, times))
            mid_loss = 0
            s_start = time.time()
        if sample_end == 0:
            end_point += (window_length - 1)
        else:
            end_point += sample_end
    stop_points.append(end_point / len(train_dataloader))
    print(stop_points[epoch])
    if (epoch + 1) % point_update_period == 0 and epoch < (point_num-1) * point_update_period:
        _, idx = torch.sort(temp_record, descending=True)
        point_array = idx[0:point_num].cpu().detach().numpy()
        max_index = point_array[0]
        last_record = point_array[0: point_num - 1]
        if max_index == window_length - 1:
            max_index = point_array[1]
        return last_record, max_index
    else:
        return 0, 0


def test(snn1, snn2, snn3, test_dataloader, criterion, device, epoch):
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
            sample_batch = x_test.shape[0]
            l1_states = snn1.create_init_states(device, sample_batch)
            l2_states = snn2.create_init_states(device, sample_batch)
            l3_states = snn3.create_init_states(device, sample_batch)
            out_spike_count = torch.zeros([sample_batch, layer3_neuron_number], device=device)
            for i in range(window_length):
                spike_l1, l1_states = snn1(x_test[:, i, :], l1_states, is_training=False)
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
        print('Average test loss {:.3f}\tAverage accuracy {:.3f}\ttime consumption for test function: {:.3}s'.format(
            test_loss, avg_acc, e - s))
        return avg_acc


def test(snn1, snn2, snn3, test_dataloader, criterion, device, epoch):
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
            sample_batch = x_test.shape[0]
            l1_states = snn1.create_init_states(device, sample_batch)
            l2_states = snn2.create_init_states(device, sample_batch)
            l3_states = snn3.create_init_states(device, sample_batch)
            out_spike_count = torch.zeros([sample_batch, layer3_neuron_number], device=device)
            for i in range(window_length):
                spike_l1, l1_states = snn1(x_test[:, i, :], l1_states, is_training=False)
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
        print('Average test loss {:.3f}\tAverage accuracy {:.3f}\ttime consumption for test function: {:.3}s'.format(
            test_loss, avg_acc, e - s))
        return avg_acc


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    snn1 = ESNN(input_size, layer1_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                 is_voltage_decay_constant=False)
    snn2 = ESNN(layer1_neuron_number, layer2_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                 is_voltage_decay_constant=False)
    snn3 = ESNN(layer2_neuron_number, layer3_neuron_number, batch_size, voltage_lambda, is_decay_constant,
                 is_voltage_decay_constant=False)
    if data_split:
        train_data = load_dataset(train_data_path, layer3_neuron_number)
        test_data = load_dataset(test_data_path, layer3_neuron_number)
    else:
        all_data = load_dataset(all_data_path, layer3_neuron_number)
        train_size = int(split_ratio * len(all_data))
        test_size = len(all_data) - train_size
        train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    params = list(snn1.parameters()) + list(snn2.parameters()) + list(snn3.parameters())
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
    for epoch in range(epochs):
        last_record, max_index = train(snn1, snn2, snn3, train_loader, criterion, optimizer, device, epoch,
                                       point_array)
        if (epoch + 1) % point_update_period == 0 and temp_index < point_num - 1:
            point_array[0:point_num - 1] = last_record
            temp_array[temp_index] = max_index
            temp_index += 1
            if temp_index == point_num - 1:
                point_array = temp_array
        acc = test(snn1, snn2, snn3, test_loader, criterion, device, epoch)
        acc_record.append(acc)
        if acc > best_acc:
            best_acc = acc
            snn1_dict = snn1.state_dict()
            snn2_dict = snn2.state_dict()
            snn3_dict = snn3.state_dict()
        print('best acc:{:.3f} '.format(best_acc))
    i = 0
    while 1:
        filename = 'SOLSA_{}_{}'.format(name, i)
        path = os.path.join(image_direct, filename)
        if not os.path.exists(path):
            os.makedirs(path)
            break
        i += 1
    snn_name1 = 'snn1_{}_{}.pkl'.format(input_size, layer1_neuron_number)
    snn_name2 = 'snn2_{}_{}.pkl'.format(layer1_neuron_number, layer2_neuron_number)
    snn_name3 = 'snn3_{}_{}.pkl'.format(layer2_neuron_number, layer3_neuron_number)
    checkpoint1 = os.path.join(path, snn_name1)
    checkpoint2 = os.path.join(path, snn_name2)
    checkpoint3 = os.path.join(path, snn_name3)
    torch.save(snn1_dict, checkpoint1)
    torch.save(snn2_dict, checkpoint2)
    torch.save(snn3_dict, checkpoint3)
    print('Successfull save model {} to {}'.format(snn_name1, checkpoint1))
    print('Successfull save model {} to {}'.format(snn_name2, checkpoint2))
    print('Successfull save model {} to {}'.format(snn_name3, checkpoint3))
    image_name = '{}_SOLSA_{}_{}_class_{:.4f}.jpg'.format(name, window_length, layer3_neuron_number, best_acc)
    image_path = os.path.join(path, image_name)
    plt.plot(epoch_list, acc_record, color='red', linewidth=2.0)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('{} SOLSA best accuracy: {}'.format(name, best_acc), fontsize='large')
    plt.savefig(image_path)
    plt.show()
    plt.close()
    saved_config_path = os.path.join(path, 'config.yaml')
    save_config(config, saved_config_path)

if __name__ == '__main__':
    main()
