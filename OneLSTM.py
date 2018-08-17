import torch
import pandas as pd
import torch.nn as nn
from torch import autograd
import torch.optim as optim
from scipy.io import loadmat
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# Center Speller
N_CHANNEL = 53                                                   # numero di canali (righe)
INPUT_SIZE = 250                                                 # features, TIME (colonne)
H_STATE = 8                                                      # stati nascosti (8/64/128)
N_LAYERS = 1                                                     # numero di livelli ricorsivi
N_CATEGORIES = 2                                                 # categorie da classificare
LEARNING_RATE = 0.01                                             # tasso di apprendimento
BATCH = 53                                                       # domanda ??????????????????????????????
BIDIRECTIONAL = False                                            # True per LSTM bidirectional
EPOCHS = 100                                                     # cicli per ogni time_series
DEVICE = torch.device('cpu')                                     # cuda per GPU
FOLDER_DATASET = '/Users/luca/Desktop/TESI/center_speller'
FILE = '/CenterSpeller_VPiac.mat_1_r_1.mat'
TRAIN_DATA = ''


class Cerebro(torch.nn.Module):
    def __init__(self, input_size, hidden_state, num_layers, bdirectional, num_categories):
        super(Cerebro, self).__init__()
        self.input_size = input_size
        self.hidden_state = hidden_state
        self.num_layer = num_layers
        self.bdirectional = bdirectional
        self.num_categories = num_categories
        # 1° livello
        self.lstm = nn.LSTM(input_size, hidden_state, num_layers, bidirectional=bdirectional)
        # 2° livello  input = n_channel * hidden state (h_size = TIME, output= n_classi)
        self.fc = nn.Linear(hidden_state, num_categories)
        # 3° livello classificatore softmax (logSoftmax per poter usare NLLoss)
        self.softmax = nn.LogSoftmax()
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch):
        if self.bdirectional:
            direction = 2
        else:
            direction = 1
        h_0 = autograd.Variable(torch.zeros(self.num_layer * direction, batch, self.hidden_state))
        c_0 = autograd.Variable(torch.zeros(self.num_layer * direction, batch, self.hidden_state))
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        # output = self.dropout_layer(hn[-1])
        output = self.dropout_layer(h_n[-1])
        output = self.fc(output)
        output = self.softmax(output)
        return output

# Caricamento File funzionante
# mat = loadmat(FOLDER_DATASET + '/CenterSpeller_VPiac.mat_1_r_1.mat')
# mat = torch.from_numpy((mat['val']))


class CerebroData(Dataset):
    def __init__(self, folder_dataset, file, size=(N_CHANNEL, INPUT_SIZE)):
        self.folder_dataset = folder_dataset
        mat = loadmat(folder_dataset + file)
        self.file = torch.from_numpy((mat['val']))
        self.size = size

    def __len__(self):
        return len(self)

    def __getitem__(self, index):
        mat = loadmat(FOLDER_DATASET + '/CenterSpeller_VPiac.mat_1_r_1.mat')
        mat = torch.from_numpy((mat['val']))
        label = self.file[index]
        return mat, label


X = Cerebro(INPUT_SIZE, H_STATE, N_LAYERS, BIDIRECTIONAL, N_CATEGORIES)#.cuda             # on GPU
optimizer = torch.optim.Adam(X.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()
train = CerebroData(FOLDER_DATASET, FILE)
train_loader = DataLoader(train, batch_size=500, shuffle=True, num_workers=1)

def train(epoch):
    X.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = X(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    print('--- Start Cerebro --- \n')
    print(X)
    for epoch in range(1, 2):
        train(epoch)

