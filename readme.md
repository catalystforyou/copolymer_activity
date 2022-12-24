# 无规多肽聚合物的酶活性能预测与优化
Term Project for the Course "Machine Learning and the Application in Chemistry"

## enviornment
- numpy
- pandas
- sklearn
- xgboost

## baseline
LinearRegression

MAE: 28.558 RMSE: 37.4457 R2: 0.1199

BayesianRidge

MAE: 28.5786 RMSE: 37.4674 R2: 0.119

MLPRegressor

MAE: 48.4744 RMSE: 59.5191 R2: -1.2308

KNeighborsRegressor

MAE: 27.8725 RMSE: 36.527 R2: 0.16

SVR

MAE: 26.8076 RMSE: 36.1233 R2: 0.1823

RandomForestRegressor

MAE: 26.8271 RMSE: 35.0135 R2: 0.2287

XGBRegressor

MAE: 29.6304 RMSE: 38.5925 R2: 0.0612

*currently there is no hyperparameter tuning here*

## results by NN
R2 score: 0.3141, RMSE: 32.52, MAE: 24.77