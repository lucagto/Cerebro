import os
import torch
import torch.nn as nn
from scipy.io import loadmat
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# Center Speller
# numero di canali (righe)
N_CHANNEL = 59
# TIME (colonne)
TIME = 800
# stati nascosti (8/64/128)
H_STATE = 8
# numero di livelli ricorsivi
N_LAYERS = 1
# categorie da classificare
N_CATEGORIES = 2
LEARNING_RATE = 0.001
BATCH = 16
# True per LSTM bidirectional
BIDIRECTIONAL = False
# cicli per ogni time_series
EPOCHS = 100
# cuda per GPU
DEVICE = torch.device('cpu')
FOLDER_DATASET = '/Users/luca/Desktop/TESI/Cerebro/Dataset/training/'
FILE = '/training/eeg59_1.mat'


class Cerebro(torch.nn.Module):
    def __init__(self, num_channel, hidden_state, num_layers, bdirectional, num_categories):
        super(Cerebro, self).__init__()
        self.num_channel = num_channel
        self.hidden_state = hidden_state
        self.num_layer = num_layers
        self.bdirectional = bdirectional
        self.num_categories = num_categories
        # 1° livello
        self.lstm = nn.LSTM(num_channel, hidden_state, num_layers, bidirectional=bdirectional, batch_first=True)
        # 2° livello  input = hidden state,  output= n_classi
        self.fc = nn.Linear(hidden_state, num_categories)
        # 3° livello classificatore softmax (logSoftmax per poter usare NLLoss)
        self.softmax = nn.LogSoftmax()
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch):
        if self.bdirectional:
            direction = 2
        else:
            direction = 1
        # h_0 = autograd.Variable(torch.zeros(self.num_layer * direction, batch, self.hidden_state))
        # c_0 = autograd.Variable(torch.zeros(self.num_layer * direction, batch, self.hidden_state))
        output = self.lstm(batch)
        output = self.dropout_layer(output[-1])
        output = self.fc(output)
        output = self.softmax(output)
        return output

# Caricamento File funzionante
# mat = loadmat(FOLDER_DATASET + '/CenterSpeller_VPiac.mat_1_r_1.mat')
# mat = torch.from_numpy((mat['val']))


class CerebroDataset(Dataset):
    def __init__(self, folder_dataset):
        super(CerebroDataset, self).__init__()
        self.folder_dataset = folder_dataset
        # lista di percorsi dei file MAT, disordinati non in ordine per nome
        self.list = os.listdir(FOLDER_DATASET)
        # size determina la lunghezza del dataset
        self.size = len(self.list)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        file_path = self.list[index]
        mat = loadmat(FOLDER_DATASET + file_path)
        mat = torch.from_numpy((mat['val']))
        label = self.list[index]
        return mat, label


X = Cerebro(N_CHANNEL, H_STATE, N_LAYERS, BIDIRECTIONAL, N_CATEGORIES)#.cuda             # on GPU
optimizer = torch.optim.Adam(X.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()
train = CerebroDataset(FOLDER_DATASET)
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0)


def train(epochs):
    X.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = X(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.data[0]))


if __name__ == '__main__':
    print('--- Start Cerebro --- \n')
    print(X)
    for epoch in range(1, 2):
        train(epoch)


