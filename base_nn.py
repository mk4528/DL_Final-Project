# -*- coding: utf-8 -*-
"""base_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16lHDKow5RkMoMrpDF8DyWXsaCL7Bf_sy
"""

import pandas as pd
payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
symbols = payload[0].Symbol

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import talib as ta
import time
import pandas_datareader.data as web
import datetime

period_y = '11y'

df_X = yf.Ticker(symbols[0]).history(period=period_y).ffill()
df_X["Close_mom_sht"] = ta.MOM(df_X["Close"], timeperiod = 5)
df_X["Close_mom_mid"] = ta.MOM(df_X["Close"], timeperiod = 25)
df_X["Close_mom_lng"] = ta.MOM(df_X["Close"], timeperiod = 75)
df_X["Close_rsi"] = ta.RSI(df_X["Close"])
df_X["Close_Price_NATR"] = ta.NATR(df_X["High"], df_X["Low"], df_X["Close"])
df_X["Close"] = ta.ROCP(df_X["Close"], timeperiod = 1)
df_X = df_X.filter(regex='Close', axis = 1)
df_X.columns = ["MMM_" + c for c in df_X.columns]

for smb in symbols[1:]:
    new_df_X = yf.Ticker(smb).history(period=period_y).ffill()
    if len(new_df_X) == 0:
        continue
    if (new_df_X.index[0] != df_X.index[0]):
        continue
    new_df_X["Close_mom_sht"] = ta.MOM(new_df_X["Close"], timeperiod = 5)
    new_df_X["Close_mom_mid"] = ta.MOM(new_df_X["Close"], timeperiod = 25)
    new_df_X["Close_mom_lng"] = ta.MOM(new_df_X["Close"], timeperiod = 75)
    new_df_X["Close_rsi"] = ta.RSI(new_df_X["Close"])
    new_df_X["Close_Price_NATR"] = ta.NATR(new_df_X["High"], new_df_X["Low"], new_df_X["Close"])
    new_df_X["Close"] = ta.ROCP(new_df_X["Close"], timeperiod = 1)
    new_df_X = new_df_X.filter(regex='Close', axis = 1)
    new_df_X.columns = [smb + "_" + c for c in new_df_X.columns]
    df_X = df_X.join(new_df_X, how = "outer")

df_X_low_cor = df_X.copy()
print(f"The shape of feature data before data reduction: {df_X_low_cor.shape}")
cor = df_X_low_cor.filter(regex='Close$', axis = 1).corr()
cor_n_diag = cor - np.diag(np.ones(cor.shape[0]))
print(f"The number of features with strong correlation (> 0.7) with another feature: {((cor_n_diag > 0.7).sum() > 0).sum()}")

while (cor_n_diag > 0.7).sum().sum() > 0:
    candidates = cor_n_diag[np.any(cor_n_diag == cor_n_diag.max().max(), axis = 1)].mean(axis = 1)
    col_del_names = candidates[candidates == candidates.max()]
    col_del_names = df_X_low_cor.filter(regex = col_del_names.index.values[0], axis = 1).columns.values

    df_X_low_cor = df_X_low_cor.loc[:, [not cl in col_del_names for cl in df_X_low_cor.columns.values]]
    cor = df_X_low_cor.filter(regex='Close$', axis = 1).corr()
    cor_n_diag = cor - np.diag(np.ones(cor.shape[0]))

df_X = df_X_low_cor
df_X = df_X.ffill()
df_X = df_X.dropna()

print(f"The feature shape after the feature selection and forward filling and droping of NA: {df_X_low_cor.shape}")

SP500 = web.DataReader(['sp500'], 'fred', df_X.index[0], datetime.datetime(2022, 5, 2))
SP500 = SP500.dropna()

SP500['sp500'] = ta.ROCP(SP500['sp500'], timeperiod = 1)
y_ind = SP500.index[1:-2]
SP500_t2 = SP500.iloc[3:, :].rename(columns = {"sp500": "sp500_t2"})
SP500_t2.index = y_ind
df = SP500_t2.join(SP500).join(df_X)

n_t = 3
df_org = df.copy()

df_ind = df_org.index[1:]
df_t_1 = df_org.iloc[:-1, 1:]
df_t_1.index = df_ind
df_t_1.columns = [c + "_t_1" for c in df_org.columns[1:]]
df = df.join(df_t_1, how = "inner")

for i in range(2, (n_t + 1)):
    df_ind = df_t_1.index[1:]
    df_t_1 = df_t_1.iloc[:-1, :]
    df_t_1.index = df_ind
    df_t_1.columns = [c + "_t_" + str(i) for c in df_org.columns[1:]]
    df = df.join(df_t_1, how = "inner")

import random
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

seed_everything(0)

mae_p = []
band_p = []
params = [["SGD", 30], ["Adagrad", 30], ["RMSprop", 30], ["Nadam", 30], ["Adadelta", 30], ["Adam", 10], ["Adam", 20], ["Adam", 30], ["Adam", 50], ["Adam", 75], ["Adam", 100], ["Adam", 200]]

for para in params:
    print("\n\noptimizer = " + para[0] + ", epoch = " + str(para[1]) + "\n")
    start = time.time()

    dropout_p = 0.1
    lr_alpha = 0.05
    n_epoch = para[1]
    n_train = int(np.round(df.shape[0] * 0.8))

    df_train = df.iloc[:n_train, :]
    df_test = df.iloc[n_train:, :]

    sc = StandardScaler()
    df_train_tf = sc.fit_transform(df_train.values)
    df_test_tf = sc.transform(df_test.values)

    NN = Sequential([Dropout(dropout_p),
                    Dense(8192, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(4096, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(2048, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(1024, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(512, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(256, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(128, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(64, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(32, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(16, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),

                    Dropout(dropout_p),
                    Dense(8, kernel_initializer = "HeUniform"),
                    LeakyReLU(alpha=lr_alpha),
                    BatchNormalization(),
                    Dropout(dropout_p),
                    Dense(1)])
    
    if para[0] == "SGD":
        NN.compile(loss = "mean_squared_error", optimizer = SGD(learning_rate = 0.001), metrics = [tf.keras.metrics.MeanAbsoluteError()])
    else:
        NN.compile(loss = "mean_squared_error", optimizer = para[0], metrics = [tf.keras.metrics.MeanAbsoluteError()])

    hist = NN.fit(x = df_train_tf[:, 1:], y = df_train_tf[:, 0], batch_size = 64, epochs = n_epoch, validation_data = (df_test_tf[:, 1:], df_test_tf[:, 0]))

    train_for_inv = np.concatenate((np.transpose([hist.history["mean_absolute_error"]]), df_train_tf[:n_epoch, 1:]), axis = 1)
    test_for_inv = np.concatenate((np.transpose([hist.history["val_mean_absolute_error"]]), df_train_tf[:n_epoch, 1:]), axis = 1)
    train_mae_org_sc = sc.inverse_transform(train_for_inv)[:, 0]
    test_mae_org_sc = sc.inverse_transform(test_for_inv)[:, 0]




    print("\n\noptimizer = " + para[0] + ", epoch = " + str(para[1]) + "\n")
    plt.figure(figsize=(20,8))
    plt.plot(range(n_epoch), hist.history["loss"], label = "train MSE");
    plt.plot(range(n_epoch), hist.history["val_loss"], label = "test MSE");
    plt.xlabel("number of epoch", size=15)
    plt.ylabel("loss", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()

    plt.figure(figsize=(20,8))
    plt.ylim([0, 2]);
    plt.plot(range(n_epoch), hist.history["loss"], label = "train MSE");
    plt.plot(range(n_epoch), hist.history["val_loss"], label = "test MSE");
    plt.xlabel("number of epoch", size=15)
    plt.ylabel("loss", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()

    plt.figure(figsize=(20,8))
    plt.plot(range(n_epoch), train_mae_org_sc * 100, label = "train MAE (original scale)");
    plt.plot(range(n_epoch), test_mae_org_sc * 100, label = "test MAE (original scale)");
    plt.xlabel("number of epoch", size=15)
    plt.ylabel("loss (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    plt.figure(figsize=(20,8))
    plt.ylim([0.2, 1.2]);
    plt.plot(range(n_epoch), train_mae_org_sc * 100, label = "train MAE (original scale)");
    plt.plot(range(n_epoch), test_mae_org_sc * 100, label = "test MAE (original scale)");
    plt.xlabel("number of epoch", size=15)
    plt.ylabel("loss (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    pred_for_inv = np.concatenate((NN.predict(df_test_tf[:, 1:]), df_test_tf[:, 1:]), axis = 1)
    pred = sc.inverse_transform(pred_for_inv)[:, 0]
    y_test = df_test.iloc[:, 0]
    
    mae_p.append(np.abs(y_test - pred).mean() * 100)
    print(f"Volatility of rate of change for S&P 500 test data: {np.std(y_test) * 100: 0.3f}%")
    print(f"Mean Absolute Error for test data (original scale): {np.abs(y_test - pred).mean() * 100: 0.3f}%")
    print(f"{(time.time() - start) / 60: 0.2f} min.")



    print("\n\noptimizer = " + para[0] + ", epoch = " + str(para[1]) + "\n")
    start = time.time()

    def predict_dist(X, model, num_samples):
        preds = [model(X, training=True) for _ in range(num_samples)]
        return np.hstack(preds)

    def predict_point(X, model, num_samples):
        pred_dist = predict_dist(X, model, num_samples)
        return pred_dist.mean(axis=1)


    pred_dist = predict_dist(df_test_tf[:, 1:], NN, 101)
    pred_mean = predict_point(df_test_tf[:, 1:], NN, 101)

    pred_dist = pd.DataFrame(pred_dist)
    pred_dist.index = df_test.index

    for i in range(pred_dist.shape[1]):
        pred_for_inv = np.concatenate((np.transpose([pred_dist.iloc[:, i]]), df_test_tf[:, 1:]), axis = 1)
        pred_dist.iloc[:, i] = sc.inverse_transform(pred_for_inv)[:, 0]

    pred_for_inv = np.concatenate((np.transpose([pred_mean]), df_test_tf[:, 1:]), axis = 1)
    pred_mean = sc.inverse_transform(pred_for_inv)[:, 0]





    plt.figure(figsize=(25,10))
    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    pred_05 = pred_dist.quantile(0.05, axis = 1)
    pred_50 = pred_dist.quantile(0.5, axis = 1)
    pred_95 = pred_dist.quantile(0.95, axis = 1)

    plt.figure(figsize=(25,10))
    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
    plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
    plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()




    pl_l = -8
    pl_u = 8


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2020, 4, 25), datetime.datetime(2020, 7, 1)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2020, 4, 25), datetime.datetime(2020, 7, 1)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
    plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
    plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()

    for i in range(3):
        plt.figure(figsize=(20,8))
        plt.xlim([datetime.datetime(2020, i * 2 + 6, 25), datetime.datetime(2020, i * 2 + 8, 25)]);
        plt.ylim(pl_l, pl_u)

        plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
        plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
        plt.xlabel("date", size=15)
        plt.ylabel("rate of change (%)", size=15)
        plt.grid()
        plt.legend(prop={'size': 15})
        plt.show()


        plt.figure(figsize=(20,8))
        plt.xlim([datetime.datetime(2020, i * 2 + 6, 25), datetime.datetime(2020, i * 2 + 8, 25)]);
        plt.ylim(pl_l, pl_u)

        plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
        plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
        plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
        plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
        plt.xlabel("date", size=15)
        plt.ylabel("rate of change (%)", size=15)
        plt.grid()
        plt.legend(prop={'size': 15})
        plt.show()

    for i in range(6):
        plt.figure(figsize=(20,8))
        if i == 0:
            plt.xlim([datetime.datetime(2020, 12, 25), datetime.datetime(2021, 2, 25)]);
        else:
            plt.xlim([datetime.datetime(2021, i * 2, 25), datetime.datetime(2021, i * 2 + 2, 25)]);
        
        plt.ylim(pl_l, pl_u)
        plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
        plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
        plt.xlabel("date", size=15)
        plt.ylabel("rate of change (%)", size=15)
        plt.grid()
        plt.legend(prop={'size': 15})
        plt.show()


        plt.figure(figsize=(20,8))
        if i == 0:
            plt.xlim([datetime.datetime(2020, 12, 25), datetime.datetime(2021, 2, 25)]);
        else:
            plt.xlim([datetime.datetime(2021, i * 2, 25), datetime.datetime(2021, i * 2 + 2, 25)]);

        plt.ylim(pl_l, pl_u)
        plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
        plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
        plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
        plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
        plt.xlabel("date", size=15)
        plt.ylabel("rate of change (%)", size=15)
        plt.grid()
        plt.legend(prop={'size': 15})
        plt.show()


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2021, 12, 25), datetime.datetime(2022, 2, 25)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2021, 12, 25), datetime.datetime(2022, 2, 25)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
    plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
    plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2022, 2, 25), datetime.datetime(2022, 5, 5)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred * 100, label = "prediction", color = "darkgreen");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    plt.figure(figsize=(20,8))
    plt.xlim([datetime.datetime(2022, 2, 25), datetime.datetime(2022, 5, 5)]);
    plt.ylim(pl_l, pl_u)

    plt.plot(df_test.index, y_test * 100, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_95 * 100, label = "prediction 95% bound", linestyle = "dashed", color = "red");
    plt.plot(df_test.index, pred_mean * 100, label = "prediction mean", linestyle = "dotted", color = "purple");
    plt.plot(df_test.index, pred_05 * 100, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
    plt.xlabel("date", size=15)
    plt.ylabel("rate of change (%)", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()







    pred_price = [100 * (pred[0] + 1)]
    y_test_price = [100 * (y_test[0] + 1)]
    for i in range(1, len(pred)):
        pred_price.append(pred_price[i-1] * (pred[i] + 1))
        y_test_price.append(y_test_price[i-1] * (y_test[i] + 1))

    plt.figure(figsize=(20,8))
    plt.plot(df_test.index, y_test_price, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_price, label = "prediction", color = "darkgreen");
    plt.xlabel("date", size=15)
    plt.ylabel("index value", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()


    pred_dist_price = (pred_dist + 1)
    pred_dist_price.iloc[0,:] = 100 * pred_dist_price.iloc[0,:]
    pred_dist_price = pred_dist_price.cumprod()

    pred_05_price = pred_dist_price.quantile(0.05, axis = 1)
    pred_50_price = pred_dist_price.quantile(0.5, axis = 1)
    pred_95_price = pred_dist_price.quantile(0.95, axis = 1)

    pred_mean_price = [100 * (pred_mean[0] + 1)]
    for i in range(1, len(pred_mean)):
        pred_mean_price.append(pred_mean_price[i-1] * (pred_mean[i] + 1))

    plt.figure(figsize=(20,8))
    plt.ylim([70, 200]);

    plt.plot(df_test.index, y_test_price, label = "S&P 500", color = "black");
    plt.plot(df_test.index, pred_95_price, label = "prediction 95% bound", linestyle = "dashed", color = "red");
    plt.plot(df_test.index, pred_mean_price, label = "prediction mean", linestyle = "dotted", color = "purple");
    plt.plot(df_test.index, pred_05_price, label = "prediction 5% bound", linestyle = "dashed", color = "blue");
    plt.xlabel("date", size=15)
    plt.ylabel("index value", size=15)
    plt.grid()
    plt.legend(prop={'size': 15})
    plt.show()

    band_p.append((pred_95 - pred_05).mean() * 100)

    print(f"{(time.time() - start) / 60: 0.2f} min.")

plt.figure(figsize=(10,10))

plt.scatter(mae_p, band_p);
plt.xlabel("Mean Absolute Error for rate of change (%, original scale)", size=15)
plt.ylabel("mean 90% band width (%)", size=15)
for i in [0, 1, 4, 5, 6]:
    plt.annotate(params[i][0] + ", " + str(params[i][1]) + " Epoch, 10% Drop", (mae_p[i] - 0.02, band_p[i] + 0.05), size = 12)
plt.grid()
plt.show()

mae_p_wo_adadelta = mae_p[2:4]
band_p_wo_adadelta = band_p[2:4]
params_wo_adadelta = params[2:4]
mae_p_wo_adadelta.extend(mae_p[6:])
band_p_wo_adadelta.extend(band_p[6:])
params_wo_adadelta.extend(params[6:])

plt.figure(figsize=(10,10))
plt.xlim(0.75, 1.05)
plt.ylim(0.4, 1.5)
plt.scatter(mae_p_wo_adadelta, band_p_wo_adadelta);
plt.xlabel("Mean Absolute Error for rate of change (%, original scale)", size=15)
plt.ylabel("mean 90% band width (%)", size=15)
for i in range(len(params_wo_adadelta)):
    plt.annotate(params_wo_adadelta[i][0] + ", " + str(params_wo_adadelta[i][1]) + " Epoch, 10% Drop", (mae_p_wo_adadelta[i] - 0.005, band_p_wo_adadelta[i] + 0.03), size = 12)
plt.grid()
plt.show()

