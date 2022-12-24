import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def add_features():
    data = pd.read_csv('data/trainval.csv')
    data['weighted_A1'] = 80 * data['A1']
    data['weighted_A2'] = 113.3 * data['A2']
    data['weighted_A3'] = 54.7 * data['A3']
    data['weighted_A4'] = 84.7 * data['A4']
    data['A1A1'] = data['A1'] * data['A1']
    data['A2A2'] = data['A2'] * data['A2']
    data['A3A3'] = data['A3'] * data['A3']
    data['A4A4'] = data['A4'] * data['A4']
    data["A1A2"] = data["A1"] * data["A2"]
    data["A1A3"] = data["A1"] * data["A3"]
    data['A1A4'] = data['A1'] * data['A4']
    data["A2A3"] = data["A2"] * data["A3"]
    data['A2A4'] = data['A2'] * data['A4']
    data['A3A4'] = data['A3'] * data['A4']
    data["A1A2A3"] = data["A1"] * data["A2"] * data["A3"]
    data['A1A2A4'] = data['A1'] * data['A2'] * data['A4']
    data['A1A3A4'] = data['A1'] * data['A3'] * data['A4']
    data['A2A3A4'] = data['A2'] * data['A3'] * data['A4']
    data['A1A2A3A4'] = data['A1'] * data['A2'] * data['A3'] * data['A4']
    data['weighted'] = 80 * data['A1'] + 113.3 * data['A2'] + 54.7 * data['A3'] + 84.7 * data['A4']
    data['weighted_A1A1'] = data['weighted_A1'] * data['weighted_A1']
    data['weighted_A2A2'] = data['weighted_A2'] * data['weighted_A2']
    data['weighted_A3A3'] = data['weighted_A3'] * data['weighted_A3']
    data['weighted_A4A4'] = data['weighted_A4'] * data['weighted_A4']
    data['weighted_A1A2'] = data['weighted_A1'] * data['weighted_A2']
    data['weighted_A1A3'] = data['weighted_A1'] * data['weighted_A3']
    data['weighted_A1A4'] = data['weighted_A1'] * data['weighted_A4']
    data['weighted_A2A3'] = data['weighted_A2'] * data['weighted_A3']
    data['weighted_A2A4'] = data['weighted_A2'] * data['weighted_A4']
    data['weighted_A3A4'] = data['weighted_A3'] * data['weighted_A4']
    data['weighted_A1A2A3'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A3']
    data['weighted_A1A2A4'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A4']
    data['weighted_A1A3A4'] = data['weighted_A1'] * data['weighted_A3'] * data['weighted_A4']
    data['weighted_A2A3A4'] = data['weighted_A2'] * data['weighted_A3'] * data['weighted_A4']
    data['weighted_A1A2A3A4'] = data['weighted_A1'] * data['weighted_A2'] * data['weighted_A3'] * data['weighted_A4']
    X = data.drop(['Activity'], axis=1).values
    data['Activity_var'] = data['Activity'] - data['weighted']
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = data['Activity'].values
    y_var = data['Activity_var'].values
    return X, y, y_var, data


def train_and_validate(X, y, y_var, data):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    MAE, RMSE, R2 = [], [], []
    for train_index, test_index in kf.split(data):
        MAE.append([])
        RMSE.append([])
        R2.append([])
        train_X, val_X = X[train_index], X[test_index]
        train_y, val_y = y[train_index], y[test_index]
        train_y_var, val_y_var = y_var[train_index], y_var[test_index]
        baseline_models = [LinearRegression(), 
                            BayesianRidge(), 
                            MLPRegressor(max_iter=100),
                            KNeighborsRegressor(n_neighbors=5), 
                            SVR(epsilon=0.2, C=1.0, gamma='auto'), 
                            RandomForestRegressor(n_estimators=500, max_depth=5, oob_score=True), 
                            XGBRegressor(n_estimators=500, max_depth=5)]
        for model in baseline_models:
            model.fit(train_X, train_y_var)
            pred = model.predict(val_X)
            MAE[-1].append(mean_absolute_error(val_y, pred+data['weighted'].values[test_index]))
            RMSE[-1].append(mean_squared_error(val_y, pred+data['weighted'].values[test_index])**0.5)
            R2[-1].append(r2_score(val_y, pred+data['weighted'].values[test_index]))
    MAE = np.array(MAE)
    RMSE = np.array(RMSE)
    R2 = np.array(R2)
    for idx, model_name in enumerate(['LinearRegression', 'BayesianRidge', 'MLPRegressor', 'KNeighborsRegressor', 'SVR', 'RandomForestRegressor', 'XGBRegressor']):
        print(model_name+'\n'+'MAE:', round(np.mean(MAE[:, idx]).item(), 4), 'RMSE:', round(np.mean(RMSE[:, idx]).item(), 4), 'R2:', round(np.mean(R2[:, idx]).item(), 4))



if __name__ == '__main__':
    X, y, y_var, data = add_features()
    train_and_validate(X, y, y_var, data)