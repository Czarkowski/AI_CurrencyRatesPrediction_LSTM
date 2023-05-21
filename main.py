import pandas as pd

import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.metrics import Precision, Recall, Accuracy

from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Bidirectional
from keras.layers import Dropout
from keras import optimizers, regularizers, Model

from tools import getDataPandas, normalize, saveData, getTimeSeriesDataBool, splitData, getInput, loadModel, \
    saveModel, plotPrint, loadData


def getData(c):
    options = ['l', 'n', 'q']
    print("Dane wejściowe:\n"
          f"{options[0]} - loadLast\n"
          f"{options[1]} - prepareNewData\n"
          f"{options[2]} - quit\n")

    x = getInput(options, c)
    if x == options[0]:
        return loadData()
    elif x == options[1]:
        return newData()
    elif x == options[2]:
        exit(0)

def scheduleLearningRate(epoch):
    initial_lr = 0.002  # Początkowa wartość learning rate
    drop = 0.5  # Współczynnik zmniejszenia learning rate
    epochs_drop = 5  # Liczba epok po których następuje zmniejszenie learning rate
    new_lr = initial_lr * np.power(drop, np.floor((epoch) / epochs_drop))
    return new_lr

    # model2 = Sequential([layers.LSTM(64, activation="relu", input_shape=(xTrain.shape[1:]), return_sequences=True),
    #                      layers.Dropout(0.2),
    #                      layers.LSTM(64, activation="relu", return_sequences=True),
    #                      layers.Flatten(),
    #                      layers.Dropout(0.2),
    #                      layers.Dense(32, activation="relu"),
    #                      layers.Dropout(0.2),
    #                      layers.Dense(32, activation="relu"),
    #                      layers.Dense(1)])


def newData():
    df = getDataPandas('EURUSD_D1')
    normalize(df)
    df = df.add_suffix('_USD', axis=1)

    dfTemp = getDataPandas('EURGBP_D1')
    normalize(dfTemp)
    dfTemp = dfTemp.add_suffix('_GBP', axis=1)
    df = pd.merge(dfTemp, df, on='Time', how='inner')

    dfTemp = getDataPandas('EURCHF_D1')
    normalize(dfTemp)

    dfTemp = dfTemp.add_suffix('_CHF', axis=1)
    df = pd.merge(dfTemp, df, on='Time', how='inner')

    df["WeekDay"] = df.index.weekday
    df = pd.get_dummies(df, columns=['WeekDay'], dtype=int)

    saveData(df)
    print(df.head())
    return df


def newModel(shape):
    model = Sequential()
    # model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(shape))))
    model.add(LSTM(64, return_sequences=True, input_shape=shape, activation='relu'))
    model.add(Dropout(0.1))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    # model.add(LSTM(64, input_shape=(xTrain.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(0.04)))
    model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=True, activation='relu'))
    model.add(Flatten())
    # model.add(LSTM(32, input_shape=(xTrain.shape[1:]), return_sequences=True))
    # model.add(LSTM(64, return_sequences=True,activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam()
    lr_scheduler = LearningRateScheduler(scheduleLearningRate)

    model.build(input_shape=(None,) + shape)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Precision(), Recall()])

    return model

defaultOptionsNew = {
        'data': 'n',
        'model': 'c',
        'weight': 't',
        'train': 'z',
        'save': 't'
    }
defaultOptionsLoad = {
        'data': 'l',
        'model': 'l',
        'weight': 'l',
        'train': 'n',
        'save': 'n'
    }
defaultOptionsLoadOnlyData = {
        'data': 'l',
        'model': 'c',
        'weight': None,
        'train': 'z',
        'save': 't'
    }
defaultOptionsNone = {
        'data': None,
        'model': None,
        'weight': None,
        'train': None,
        'save': None
    }
def main():
    defaultOptions = defaultOptionsLoadOnlyData
    df = getData(defaultOptions['data'])
    timeSeriesTuple = getTimeSeriesDataBool(df.to_numpy(), df.columns.get_loc('Close_USD'), 30, 5, 1)
    trainData, testData = splitData(timeSeriesTuple, 0.1)
    dataShape = trainData[0].shape[1:]
    print(f'\nData shape: {dataShape}\n')

    options = ['l', 'c', 'q']
    model: Model
    print(f"{options[0]} - loadModel\n"
          f"{options[1]} - createNewAndTrain\n"
          f"{options[2]} - guit\n")
    x = getInput(options, defaultOptions['model'])
    if x == options[0]:
        model = loadModel(defaultOptions['weight'])
    elif x == options[1]:
        model = newModel(dataShape)
    elif x == options[2]:
        exit(0)
    print(model.summary())

    if True:
        options = ['z', 'b', 'n']
        print('Trenuj model:\n'
              f"{options[0]} - zWalidacją\n"
              f"{options[1]} - bezWalidacji\n"
              f"{options[2]} - NieTrenuj\n")
        x = getInput(options, defaultOptions['train'])
        valid = None
        if x != options[2]:
            if x == options[0]:
                valid = testData
            BATCH_SIZE = 32
            EPOCHS = 40
            lrScheduler = LearningRateScheduler(scheduleLearningRate)
            history = model.fit(trainData[0], trainData[1], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=valid,
                                callbacks=[lrScheduler])

            if x == options[0]:
                plotPrint(history)


    trainPre = model.predict(trainData[0])
    testPre = model.predict(testData[0])

    pred = trainPre.reshape(-1)
    predBool = (pred >= 0.5).astype(int)
    accTrain = (predBool == trainData[1]).astype(int)
    count_ones = np.count_nonzero(accTrain == 1)
    acc = (count_ones / len(accTrain) * 100)
    print(f'Celnosc na zbiorze uczącym = {acc}')

    pred = testPre.reshape(-1)
    threshold = 0.50
    # predBool = (pred >= 0.5).astype(int)
    predBool = np.where(pred > threshold, 1, 0)
    print(f'Tablica predykcji:\n{predBool}')
    accTrain = (predBool == testData[1]).astype(int)
    count_ones = np.count_nonzero(accTrain == 1)
    acc = (count_ones / len(accTrain) * 100)
    print(f'Celnosc na zbiorze testowym/validacyjnym = {acc}')

    length = len(testPre)
    # Prawdziwie Pozytywny
    PP = np.count_nonzero(np.logical_and((predBool == 1), (testData[1] == 1)).astype(int)) / length * 100
    print(f'Prawdziwie pozytywny: {PP}')

    # Prawdziwie Negatywny
    PN = np.count_nonzero(np.logical_and((predBool == 0), (testData[1] == 0)).astype(int)) / length * 100
    print(f'Prawdziwie negatywny: {PN}')

    # Fałszywie Pozytywny
    FP = np.count_nonzero(np.logical_and((predBool == 1), (testData[1] == 0)).astype(int)) / length * 100
    print(f'Fałszywie pozytywny: {FP}')

    # Fałszywie Newgatywny
    FN = np.count_nonzero(np.logical_and((predBool == 0), (testData[1] == 1)).astype(int)) / length * 100
    print(f'Fałszywie negatywny: {FN}')

    options = ['t', 'n', 'q']
    print('Zapisać model?\n'
          f"{options[0]} - tak\n"
          f"{options[1]} - nie\n"
          f"{options[2]} - guit\n")
    x = getInput(options, defaultOptions['save'])
    if x == 't':
        saveModel(model)
    elif x == options[2]:
        exit(0)


if __name__ == "__main__":
    main()
