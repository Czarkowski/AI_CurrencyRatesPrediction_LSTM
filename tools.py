import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from keras.saving.saving_api import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


runDataTime = datetime.now()
modelsDir = 'NeuralNetwork'
datasDir = 'PrepareData'


def getInput(valid, x=None):
    while True:
        if x in valid:
            return x
        x = input().lower()




def getDataPandas(name: str = 'EURUSD_D1'):
    colNames = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(f'DataLSTM/{name}.csv', index_col=False, names=colNames, date_format='%Y-%m-%d %H:%M',
                     parse_dates=['Time'])
    df["WeekDay"] = df['Time'].dt.weekday

    # połaczenie niedzieli z poniedziałkiem
    for i, row in df.iterrows():
        if row['WeekDay'] == 6:
            if i + 1 >= df.shape[0]:
                print('out of range!')
                continue
            df.loc[i + 1, "Open"] = row['Open']
            df.loc[i + 1, "High"] = max(row['High'], df.loc[i + 1, "High"])
            df.loc[i + 1, "Low"] = min(row['Low'], df.loc[i + 1, "Low"])
            df.loc[i + 1, "Volume"] = sum([row['Volume'], df.loc[i + 1, "Volume"]])

    df = df[df.WeekDay != 6]
    df.reset_index(inplace=True, drop=True)

    # w niektóre piątki giełda nie była otwarta, naiwnie uzupełniamy brakujące dni danymi z dnia poprzedniego
    for i, row in df.iterrows():
        if row.WeekDay == 3:
            if df.loc[i + 1, 'WeekDay'] != 4:
                print(row.Time)
                row.WeekDay = 4
                row.Time += timedelta(days=1)
                df = pd.concat([df, pd.DataFrame([row])])

    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)

    # df['WeekDay'] = df['WeekDay'].astype(int)
    df.drop(['WeekDay'], inplace=True, axis=1)

    return df



def getTimeSeriesDataBool(tab: np.ndarray, targetIndes: int, stepBack: int = 1, stepForward: int = 1,
                          stepNext: int = 1) -> tuple[list[list[None]], list[None]]:
    listNew = []
    listValue = []
    start = stepBack - 1
    for j in range(start, tab.shape[0] - stepForward, stepNext):
        listTemp = []
        for k in range(0, stepBack):
            listTemp.append(tab[j + (k - start)])
        listNew.append(listTemp)
        listValue.append((tab[j + stepForward][targetIndes] >= tab[j][targetIndes]).astype(int))
    return np.array(listNew), np.array(listValue)


def getTimeSeriesData(tab: np.ndarray, targetIndes: int, stepBack: int = 1, stepForward: int = 1,
                      stepNext: int = 1) -> tuple[list[list[None]], list[None]]:
    listNew = []
    listValue = []
    start = stepBack - 1
    for j in range(start, tab.shape[0] - stepForward, stepNext):
        if math.isnan(tab[j + stepForward][targetIndes]):
            print(f'index {j} has target value {tab[j + stepForward][targetIndes]}')
            continue
        if math.isnan(tab[j][targetIndes]):
            print(f'index {j} has target value {tab[j + stepForward][targetIndes]}')
            continue
        listTemp = []
        for k in range(0, stepBack):
            listTemp.append(tab[j + (k - start)])
        listNew.append(listTemp)
        listValue.append(tab[j + stepForward][targetIndes])
    return (listNew, listValue)


def splitData(ts, size):
    xTrain, xTest, yTrain, yTest = train_test_split(ts[0], ts[1], test_size=size, shuffle=True)
    return (xTrain, yTrain), (xTest, yTest)

def normalize(df):
    meanPrice = (df.loc[:, ['Open', 'High', 'Low', 'Close']].mean()).mean()
    df[['Open', 'High', 'Low', 'Close']] -= meanPrice

    meanVolume = (df['Volume'].mean())
    df.Volume -= meanVolume
    stdVolume = (df.Volume.std())
    df.Volume /= stdVolume
    return df

def saveModel(model):
    name = runDataTime.strftime("%Y-%m-%d_%H_%M")
    print(f'Save model: {name}')
    model.save(f'{modelsDir}/{name}/model.h5')
    model.save_weights(f'{modelsDir}/{name}/weights.h5')
    f = open(f'{modelsDir}/last.txt', "w")
    f.write(name)
    f.close()


def loadModel():
    print("loadModel")
    f = open(f"{modelsDir}/last.txt", "r")
    name = f.readline()
    print(f'model name: {name}')
    f.close()
    model = load_model(f'{modelsDir}/{name}/model.h5')
    options = ['l', 't', 'q']
    print(f"{options[0]} - loadWeights\n"
          f"{options[1]} - trainModel\n"
          f"{options[2]} - guit\n")
    x = getInput(options)
    if x == options[0]:
        model.load_weights(f'{modelsDir}/{name}/weights.h5')
    elif x == options[2]:
        exit(0)
    return model


def saveData(df):
    name = runDataTime.strftime("%Y-%m-%d_%H_%M")
    df.to_csv(f'{datasDir}/{name}' + '.csv', float_format='%.4f')
    f = open(f"{datasDir}/last.txt", "w")
    f.write(name)
    f.close()

def loadData():
    print("load")
    f = open(f"{datasDir}/last.txt", "r")
    name = f.readline()
    f.close()
    df = pd.read_csv(f'{datasDir}/{name}.csv', index_col='Time', parse_dates=['Time'])
    print(df.head())
    return df


def plotPrint(history):

    epochs = np.arange(len(history.history['val_loss'])) + 1
    fig = plt.figure(figsize=(8, 4))
    if 'accuracy' in history.history:
        ax1 = fig.add_subplot(121)
        ax1.plot(epochs, history.history['loss'], c='b', label='Train loss')
        ax1.plot(epochs, history.history['val_loss'], c='g', label='Valid loss')
        plt.legend(loc='lower left');
        plt.grid(True)

        ax1 = fig.add_subplot(122)
        ax1.plot(epochs, history.history['accuracy'], c='b', label='Train acc')
        ax1.plot(epochs, history.history['val_accuracy'], c='g', label='Valid acc')
        plt.legend(loc='lower right');
        plt.grid(True)


    else:
        ax1 = fig.add_subplot(111)
        ax1.plot(epochs, history.history['loss'], c='b', label='Train loss')
        ax1.plot(epochs, history.history['val_loss'], c='g', label='Valid loss')
        plt.legend(loc='lower left');
        plt.grid(True)
    plt.show()

