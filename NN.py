import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_data(name='trainval'): # 简单设计了一下feature，重点在引入了weighted
    data = pd.read_csv(f'data/{name}.csv')
    data['weighted_A1'] = 80 * data['A1']
    data['weighted_A2'] = 113.3 * data['A2']
    data['weighted_A3'] = 54.7 * data['A3']
    data['weighted_A4'] = 84.7 * data['A4']
    data['weighted'] = 80 * data['A1'] + 113.3 * data['A2'] + 54.7 * data['A3'] + 84.7 * data['A4']
    if name == 'trainval':
        X = data.drop(['Activity'], axis=1).values
        data['Activity_var'] = data['Activity'] - data['weighted']
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        y = data['Activity'].values
        y_var = data['Activity_var'].values
        return X, y, y_var, data
    else:
        data_ = pd.read_csv(f'data/trainval.csv')
        data_['weighted_A1'] = 80 * data_['A1']
        data_['weighted_A2'] = 113.3 * data_['A2']
        data_['weighted_A3'] = 54.7 * data_['A3']
        data_['weighted_A4'] = 84.7 * data_['A4']
        data_['weighted'] = 80 * data_['A1'] + 113.3 * data_['A2'] + 54.7 * data_['A3'] + 84.7 * data_['A4']
        X_ = data_.drop(['Activity'], axis=1).values
        data_['Activity_var'] = data_['Activity'] - data_['weighted']
        scaler = StandardScaler()
        scaler.fit(X_)
        X = data.values
        X = scaler.transform(X)
        return X, data

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
    def __init__(self, input_size, hidden_size=512, activation1='tanh', activation2='tanh', dropout=0.2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activate1 = torch.tanh if activation1 == 'tanh' else (torch.relu if activation1 == 'relu' else torch.sigmoid)
        self.activate2 = torch.tanh if activation2 == 'tanh' else (torch.relu if activation1 == 'relu' else torch.sigmoid)
        self.dropout = nn.Dropout(dropout)
        self.CNN_model = CNN()
    def forward(self, X, random=None):
        # feat = self.CNN_model(random) # 这里指望它能提取一些特征，但效果不好
        X = self.activate1(self.fc1(X))
        X = self.dropout(X)
        X = self.activate2(self.fc2(X))
        X = self.dropout(X)
        # X = torch.cat((X, feat), dim=1)
        X = self.fc3(X)
        return X

def train(model, X_train, y_train, X_val, y_val, y_true, y_ref, random_X_train, random_X_val, patience=100, epochs=1000, lr=0.0003, opt='Adam', weight_decay=1e-5, verbose=True, tuning=False):
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loss_train = []
    loss_val = []
    if not tuning:
        best_epoch, best_R2, best_MAE, best_RMSE = 0, 0.3, 0, 0
    else:
        best_epoch, best_R2, best_MAE, best_RMSE = 0, 0, 0, 0
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
            y_pred = y_pred.reshape(-1).detach().cpu().numpy()
            try:
                curr_R2 = r2_score(y_true, y_pred+y_ref)
            except Exception as e:
                curr_R2 = 0
            if curr_R2 > best_R2:
                best_R2 = r2_score(y_true, y_pred+y_ref)
                best_MAE = mean_absolute_error(y_true, y_pred+y_ref)
                best_RMSE = np.sqrt(mean_squared_error(y_true, y_pred+y_ref))
                best_epoch = epoch
                if epoch - best_epoch > patience and best_epoch > 0:
                    break
                if not tuning:
                    torch.save(model.state_dict(), 'ckpts/best_model.pth')
        if epoch % 10 == 0 and verbose:
            print(f'Epoch: {epoch}, Train loss: {loss_train[-1]}, Val loss: {loss_val[-1]}')
            print(f'R2 score: {r2_score(y_true, y_pred+y_ref)}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred+y_ref))}, MAE: {mean_absolute_error(y_true, y_pred+y_ref)}')
    return best_R2, best_MAE, best_RMSE, best_epoch

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
    model = NN(X_train.shape[1], hidden_size=512, dropout=0.05, activation1='tanh', activation2='tanh')
    # model = CNN()
    best_R2, best_MAE, best_RMSE, best_epoch = train(model, X_train, y_train, X_val, y_val, y_true, y_ref, random_X_train, random_X_val, epochs=1000, lr=0.01, weight_decay=1e-5, opt='Adam')
    print(f'Best R2 score: {best_R2}, Best RMSE: {best_RMSE}, Best MAE: {best_MAE}, Best epoch: {best_epoch}')
    # loss_train, loss_val = train_orig(model, X_train, y_true_train, X_val, y_val, y_true, y_ref, epochs=1000)

def tuning():
    import optuna
    from optuna.samplers import TPESampler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, y_var, data = load_data() # variance对应的是activity - weight*x的结果
    random, _ = gen_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y_var, test_size=0.2, random_state=42)
    random_X_train, random_X_val = train_test_split(random, test_size=0.2, random_state=42)
    _, _, y_true_train, y_true = train_test_split(X, data['Activity'].values, test_size=0.2, random_state=42)
    _, _, _, y_ref = train_test_split(X, data['weighted'].values, test_size=0.2, random_state=42)
    X_train = torch.from_numpy(X_train).float().to(device)
    random_X_train = torch.from_numpy(random_X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    random_X_val = torch.from_numpy(random_X_val).float().to(device)
    y_train = torch.from_numpy(y_train).float().reshape(-1, 1).to(device)
    y_true_train = torch.from_numpy(y_true_train).float().reshape(-1, 1).to(device)
    y_val = torch.from_numpy(y_val).float().reshape(-1, 1).to(device)
    def objective(trial):
        best_R2s = []
        for i in range(5):
            hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
            activation1 = trial.suggest_categorical("activation1", ["relu", "tanh", "sigmoid"])
            activation2 = trial.suggest_categorical("activation2", ["relu", "tanh", "sigmoid"])
            dropout = trial.suggest_categorical("dropout", [0, 0.05, 0.1, 0.2])
            model = NN(input_size=X_train.shape[1], hidden_size=hidden_size, activation1=activation1, activation2=activation2, dropout=dropout)
            model.to(device)
            lr = trial.suggest_categorical("lr", [1e-2, 3e-3, 1e-3, 3e-4])
            weight_decay = trial.suggest_categorical("weight_decay", [3e-5, 1e-5, 3e-6])
            opt = trial.suggest_categorical("opt", ["Adam", "SGD", 'RMSprop'])
            best_R2, best_MAE, best_RMSE, best_epoch = train(model, X_train, y_train, X_val, y_val, y_true, y_ref, random_X_train, random_X_val, epochs=1000, 
                                                            lr=lr, weight_decay=weight_decay, opt=opt, verbose=False, tuning=True)   
            best_R2s.append(best_R2)         
        '''trial.report(np.mean(best_R2s), i)
        if trial.should_prune():
            raise optuna.TrialPruned()'''
        return np.mean(best_R2s)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=2000)
    return study.best_params

def test():
    X, data = load_data('test')
    model = NN(X.shape[1], hidden_size=512, dropout=0.05, activation1='tanh', activation2='tanh')
    model.load_state_dict(torch.load('ckpts/best_model.pth'))
    model.eval()
    X = torch.from_numpy(X).float()
    y_pred = model(X)
    y_pred = y_pred.detach().numpy().reshape(-1)
    data['Activity'] = y_pred + data['weighted'].values
    data.drop(['weighted', 'weighted_A1', 'weighted_A2', 'weighted_A3', 'weighted_A4'], axis=1, inplace=True)
    data.to_csv('data/result.csv', index=False)

if __name__ == '__main__':
    # best_params = tuning()
    # print(best_params)
    # main()
    test()
