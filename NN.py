import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(): # 简单设计了一下feature，重点在引入了weighted
    data = pd.read_csv('data/trainval.csv')
    data['weighted_A1'] = 80 * data['A1']
    data['weighted_A2'] = 113.3 * data['A2']
    data['weighted_A3'] = 54.7 * data['A3']
    data['weighted_A4'] = 84.7 * data['A4']
    data['weighted'] = 80 * data['A1'] + 113.3 * data['A2'] + 54.7 * data['A3'] + 84.7 * data['A4']
    X = data.drop(['Activity'], axis=1).values
    data['Activity_var'] = data['Activity'] - data['weighted']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = data['Activity'].values
    y_var = data['Activity_var'].values
    return X, y, y_var, data

def gen_data(): # 产生随机序列，用于CNN采样
    data = pd.read_csv('data/trainval.csv')
    random = np.random.random(size=(data.shape[0], 64, 64))
    for r, x1, x2, x3 in zip(random, data['A1'], data['A2'], data['A3']):
        r[r>x1+x2+x3] = 84.7
        r[(r>x1+x2) & (r<=x1+x2+x3)] = 54.7
        r[(r>x1) & (r<=x1+x2)] = 113.3
        r[r<=x1] = 80
    # print(random)
    return random, data



class CNN(nn.Module): # CNN模型，有2D和1D两种，目前只用了2D，可以都用（但效果也不会变好）
    def __init__(self, hidden_size=1024):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool2d_1 = nn.MaxPool2d(2, 2)
        self.conv2d_2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2d_2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*16*16, hidden_size)
        self.conv1d_1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool1d_1 = nn.MaxPool1d(2, 2)
        self.conv1d_2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool1d_2 = nn.MaxPool1d(2, 2)
        self.fc2 = nn.Linear(32*64*16, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//8, 1)

    def forward(self, X):
        X = X.reshape(-1, 1, 64, 64)
        X_2d = F.relu(self.conv2d_1(X))
        X_2d = self.pool2d_1(X_2d)
        X_2d = F.relu(self.conv2d_2(X_2d))
        X_2d = self.pool2d_2(X_2d)
        X_2d = F.relu(self.fc1(X_2d.reshape(-1, 32*16*16)))
        '''
        X_1d = F.relu(self.conv1d_1(X.reshape(-1, 1, 64*64)))
        X_1d = self.pool1d_1(X_1d)
        X_1d = F.relu(self.conv1d_2(X_1d))
        X_1d = self.pool1d_2(X_1d)
        X_1d = F.relu(self.fc2(X_1d.reshape(-1, 32*64*16)))'''
        X = X_2d#torch.cat((X_2d, X_1d), dim=1)
        X = torch.tanh(self.fc3(X))
        # X = self.fc4(X)
        return X

class NN(nn.Module): # 用来拟合的神经网络
    def __init__(self, input_size, hidden_size=512):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.CNN_model = CNN()
    def forward(self, X, random):
        # feat = self.CNN_model(random) # 这里指望它能提取一些特征，但效果不好
        X = torch.tanh(self.fc1(X))
        X = self.dropout(X)
        X = torch.tanh(self.fc2(X))
        X = self.dropout(X)
        # X = torch.cat((X, feat), dim=1)
        X = self.fc3(X)
        return X

def train(model, X_train, y_train, X_val, y_val, y_true, y_ref, random_X_train, random_X_val, epochs=100, lr=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train, random_X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val, random_X_val)
            loss = loss_fn(y_pred, y_val)
            loss_val.append(loss.item())
            y_pred = y_pred.reshape(-1).detach().numpy()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train loss: {loss_train[-1]}, Val loss: {loss_val[-1]}')
            print(f'R2 score: {r2_score(y_true, y_pred+y_ref)}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred+y_ref))}, MAE: {mean_absolute_error(y_true, y_pred+y_ref)}')
    return loss_train, loss_val

def train_orig(model, X_train, y_train, X_val, y_val, y_true, y_ref, epochs=100, lr=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            loss_val.append(loss.item())
            y_pred = y_pred.reshape(-1).detach().numpy()
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Train loss: {loss_train[-1]}, Val loss: {loss_val[-1]}')
            print(f'R2 score: {r2_score(y_true, y_pred)}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}, MAE: {mean_absolute_error(y_true, y_pred)}')
    return loss_train, loss_val

def train_CNN(model, X_train, y_train, X_val, y_val, y_true, y_ref, epochs=100, lr=0.0003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val)
            loss = loss_fn(y_pred, y_val)
            loss_val.append(loss.item())
            y_pred = y_pred.reshape(-1).detach().numpy()
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Train loss: {loss_train[-1]}, Val loss: {loss_val[-1]}')
            print(f'R2 score: {r2_score(y_true, y_pred)}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}, MAE: {mean_absolute_error(y_true, y_pred)}')
    return loss_train, loss_val

def main():
    X, y, y_var, data = load_data() # variance对应的是activity - weight*x的结果
    random, _ = gen_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y_var, test_size=0.2, random_state=42)
    random_X_train, random_X_val = train_test_split(random, test_size=0.2, random_state=42)
    _, _, y_true_train, y_true = train_test_split(X, data['Activity'].values, test_size=0.2, random_state=42)
    _, _, _, y_ref = train_test_split(X, data['weighted'].values, test_size=0.2, random_state=42)
    X_train = torch.from_numpy(X_train).float()
    random_X_train = torch.from_numpy(random_X_train).float()
    X_val = torch.from_numpy(X_val).float()
    random_X_val = torch.from_numpy(random_X_val).float()
    y_train = torch.from_numpy(y_train).float().reshape(-1, 1)
    y_true_train = torch.from_numpy(y_true_train).float().reshape(-1, 1)
    y_val = torch.from_numpy(y_val).float().reshape(-1, 1)
    model = NN(X_train.shape[1])
    # model = CNN()
    loss_train, loss_val = train(model, X_train, y_train, X_val, y_val, y_true, y_ref, random_X_train, random_X_val, epochs=1000)
    # loss_train, loss_val = train_orig(model, X_train, y_true_train, X_val, y_val, y_true, y_ref, epochs=1000)


if __name__ == '__main__':
    main()
    # gen_data()
