from utils import *
from torch.utils.data import Dataset, DataLoader
from torch import nn
from deal import load_config
import os
import time

config = load_config('res_config.yaml')
dataset_index = '4'
dataset_config = config[dataset_index]
for key, value in dataset_config.items():
    print(f"{key}: {value}")
name = dataset_config['name']
input_size = dataset_config['input_size']
layer1_neuron_number = dataset_config['layer1']
layer2_neuron_number = dataset_config['layer2']
layer3_neuron_number = dataset_config['layer3']
window_length = dataset_config['window_length']

# to do experiment simulating different sample-rate input, change the corresponding train and test windows length
train_window_length = window_length  # default train window length is the original window length of the dataset
test_window_length = window_length  # default test window length is the original window length of the dataset

batch_size = 1  # default online learning setting
whether_mse = True
epochs = 100
learning_rate = 0.0001

train_data_path = 'dataset/{}/dealed_data/train_data.npz'.format(name)
test_data_path = 'dataset/{}/dealed_data/test_data.npz'.format(name)
image_direct = 'dataset/{}/result/'.format(name)


def train(lstm, train_dataloader, criterion, optimizer, device, epoch):
    lstm.train()
    lstm.to(device)
    s_start = time.time()
    samples = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        x_train = sample_batched[0].to(device)
        cross_target = sample_batched[1].to(device)
        mse_target = sample_batched[2].to(device)
        samples += x_train.shape[0]
        output = lstm(x_train)
        if whether_mse:
            loss = criterion(output, mse_target * train_window_length)
        else:
            loss = criterion(output, cross_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i_batch + 1) % 20 == 0:
            s_end = time.time()
            times = s_end - s_start
            print(
                "Training epoch{}/{}: [{}/{}]\tLoss: {} \tTime consumption: {:.8}s ".format(
                    epoch, epochs, samples, len(train_dataloader), loss.item(), times))
            s_start = time.time()


def test(lstm, test_dataloader, criterion, device, epoch):
    lstm.eval()
    test_loss = 0
    acc = 0
    s = time.time()
    samples = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dataloader):
            x_test = sample_batched[0].to(device)
            cross_target = sample_batched[1].to(device)
            mse_target = sample_batched[2].to(device)
            output = lstm(x_test)
            if whether_mse:
                loss = criterion(output, mse_target)
            else:
                loss = criterion(output, cross_target.long())
            test_loss += loss
            max_index = torch.argmax(output, dim=1, keepdim=False)
            correct_index = max_index == cross_target
            acc += sum(correct_index)
            samples += x_test.shape[0]
        test_loss = test_loss.item()
        test_loss /= len(test_dataloader)
        avg_acc = acc.item() / samples
        e = time.time()
        print(
            'Testing epoch{}/{} Average test loss {:.6f}\tAverage accuracy {:.6f}\ttime consumption for test function: {:.8}s'.format(
                epoch, epochs, test_loss, avg_acc, e - s))
        return avg_acc


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # lstm_size: 40 for LSTM_low, 45 for LSTM_high(when compare with SNN hidden size 100)
    lstm = LSTM_Baseline(input_size, 40, layer3_neuron_number)

    train_data = MTSCDataset(train_data_path, layer3_neuron_number, original_window_length=window_length,
                             rated_window_length=train_window_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_data = MTSCDataset(test_data_path, layer3_neuron_number, original_window_length=window_length,
                            rated_window_length=test_window_length)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False)
    params = list(lstm.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    if whether_mse:
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    acc_record = []
    epoch_list = np.arange(epochs).tolist()
    lstm_dict = {}

    for epoch in range(epochs):
        train(lstm, train_loader, criterion, optimizer, device, epoch)
        acc = test(lstm, test_loader, criterion, device, epoch)
        acc_record.append(acc)
        if acc > best_acc:
            best_acc = acc
            lstm_dict = lstm.state_dict()
        print('best acc:{}'.format(best_acc))

    # make directory for new experiment results
    i = 0
    path = ""
    while 1:
        model_directory = '{}_LSTM_baseline_{}'.format(name, i)
        path = os.path.join(result_path, model_directory)
        if not os.path.exists(path):
            os.makedirs(path)
            break
        i += 1
    model_name = 'LSTM_baseline_{}'.format(name)
    checkpoint = os.path.join(path, model_name)
    torch.save(lstm_dict, checkpoint)
    print('Successfull save model {} to {}'.format(model_name, checkpoint))


if __name__ == '__main__':
    main()
