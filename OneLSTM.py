import os
import time
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

N_CHANNEL = 59                  # numero di canali (righe)
TIME = 800                      # campioni per ogni istante di tempo (colonne)
H_STATE = 8                    # stati nascosti (8/16/32/64/128...)
N_LAYERS = 1                    # numero di livelli ricorsivi
N_CATEGORIES = 2                # categorie da classificare
LEARNING_RATE = 0.001
BATCH = 8
EPOCH = 15
BIDIRECTIONAL = False           # True per LSTM bidirectional
TRAINIG_FOLDER = '/Users/luca/Desktop/TESI/Cerebro/Dataset/training/'
TEST_FOLDER = '/Users/luca/Desktop/TESI/Cerebro/Dataset/test/'
SAVE = '/Users/luca/Desktop/TESI/Cerebro/model'
RISULTATI = '/Users/luca/Desktop/TESI/Cerebro/risultati.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Funzione per produrre un grafico della funzione di perdita e dell'accuratezza,
# prendo soltanto l'ultimo valore assunto per ogni epoca
def draw(loss_train, loss_validation, accuracy_validation):
    x_arr = range(0, EPOCH)
    y_loss_t = loss_train
    y_loss_v = loss_validation
    y_acc_v = accuracy_validation
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.12, 0.8, 0.8])
    ax.set_ylim(0, 2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss - Accuracy')

    ax.plot(x_arr, y_loss_t, color='red', LW=1, label='loss training')
    ax.plot(x_arr, y_loss_v, color='green', LW=1, label='loss test')
    ax.plot(x_arr, y_acc_v, color='blue', LW=1, label='accuracy test')
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.show()
    # salvo il grafico con nome generato dai parametri del modello
    if BIDIRECTIONAL:
        name_file = 'H_' + repr(H_STATE) + '_B_' + repr(BATCH) + '_EPO_' + repr(EPOCH) + '_BI' + '.png'
    else:
        name_file = 'H_' + repr(H_STATE) + '_B_' + repr(BATCH) + '_EPO_' + repr(EPOCH) + '_ONE' + '.png'
    fig.savefig(name_file)

# Funzione per tenere traccio del tempo di esecuzione
class Timer:
    def __init__(self):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def get_time_hhmmss(self):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str


class Cerebro(torch.nn.Module):
    def __init__(self, num_channel, hidden_state, num_layers, bd, num_categories):
        super(Cerebro, self).__init__()
        self.num_channel = num_channel
        self.hidden_state = hidden_state
        self.num_layer = num_layers
        self.bd = bd
        self.num_categories = num_categories
        self.lstm = nn.LSTM(num_channel, hidden_state, num_layers, bidirectional=bd, batch_first=True)
        if bd:
            self.fc = nn.Linear(hidden_state*2, num_categories)
        else:
            self.fc = nn.Linear(hidden_state, num_categories)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch):
        output, _ = self.lstm(batch)
        output = output[:, -1, :]
        output = self.dropout_layer(output)
        output = self.fc(output)
        output = self.softmax(output)
        return output


class CerebroDataset(Dataset):
    def __init__(self, folder_dataset):
        super(CerebroDataset, self).__init__()
        self.dataset = folder_dataset
        self.list = os.listdir(folder_dataset)
        self.size = len(self.list)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        file_path = os.path.join(self.dataset + self.list[index])
        mat = loadmat(file_path)
        val = torch.from_numpy(mat['val']).float()
        label = mat['label'][0][0]
        return val, label


model = Cerebro(N_CHANNEL, H_STATE, N_LAYERS, BIDIRECTIONAL, N_CATEGORIES)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train = CerebroDataset(TRAINIG_FOLDER)
test = CerebroDataset(TEST_FOLDER)
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0)
test_loader = DataLoader(test, batch_size=BATCH, shuffle=True, num_workers=0)
criterion = nn.CrossEntropyLoss()


def train(epochs):
    model.train()
    sum_loss = 0.0
    num_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader, 0):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        labels = labels.long()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        sum_loss += loss_value
        num_loss += 1

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                sum_loss/num_loss))
    return sum_loss/num_loss


def validation(epochs):
    model.eval()
    sum_loss = 0.0
    num_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader, 0):
        data, labels = data.to(device), labels.to(device)
        labels = labels.long()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss_value = loss.item()
        sum_loss += loss_value
        num_loss += 1
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader),
                sum_loss/num_loss) + '   Test Accuracy: %d %%' % (100 * correct/total))
    return sum_loss/num_loss, correct/total


if __name__ == '__main__':
    my_timer = Timer()
    print('--- Starting Cerebro --- \n')
    print(model)
    plot_t = []
    plot_v = []
    accuracy_v = []

    for epoch in range(EPOCH):
        print('\n---------------------------------------------------')
        loss_t = train(epoch)

        plot_t.append(loss_t)
        print()

        loss_v, accuracy = validation(epoch)

        plot_v.append(loss_v)
        accuracy_v.append(accuracy)

    draw(plot_t, plot_v, accuracy_v)
    torch.save(model.state_dict(), SAVE)
    time_hhmmss = my_timer.get_time_hhmmss()
    print("Tempo di esecuzione: %s" % time_hhmmss)
    out_file = open(RISULTATI, "a")

    if BIDIRECTIONAL:
        STRINGA = 'LSTM Bidirectional\n' + 'Hidden State: ' + repr(H_STATE) + \
                  '\nBatch Size: ' + repr(BATCH) + '\nEpochs: ' + repr(EPOCH) + \
                  '\nTraining Loss: ' + "{0:.4f}".format(plot_t[-1]) + '\nTest Loss:' \
                  + "{0:.4f}".format(plot_v[-1]) + '\nAccuracy :' + "{0:.2f}".format(100*accuracy_v[-1]) + '%'\
                  + "\nTempo di esecuzione: " + time_hhmmss
    else:
        STRINGA = 'LSTM OneDirection\n' + 'Hidden State: ' + repr(H_STATE) + \
                  '\nBatch Size: ' + repr(BATCH) + '\nEpochs: ' + repr(EPOCH) + \
                  '\nTraining Loss: ' + "{0:.4f}".format(plot_t[-1]) + '\nTest Loss:' \
                  + "{0:.4f}".format(plot_v[-1]) + '\nAccuracy :' + "{0:.2f}".format(100*accuracy_v[-1]) + '%'\
                  + "\nTempo di esecuzione: " + time_hhmmss

    out_file.write("\n-------------------------------------------------\n")
    out_file.write(STRINGA)
    out_file.close()



