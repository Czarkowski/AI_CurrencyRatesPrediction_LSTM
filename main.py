import pandas as pd
import tensorflow as tf
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
    initial_lr = 0.0015  # Początkowa wartość learning rate
    drop = 0.8  # Współczynnik zmniejszenia learning rate
    epochs_drop = 5  # Liczba epok po których następuje zmniejszenie learning rate
    new_lr = initial_lr * np.power(drop, np.floor((epoch+2) / epochs_drop))
    return new_lr



def newData():
    dic = dict()
    df = getDataPandas('EURUSD_D1')
    dic['EURUSD'] = normalize(df)
    df = df.add_suffix('_USD', axis=1)

    dfTemp = getDataPandas('EURGBP_D1')
    dic['EURGBP'] = normalize(dfTemp)
    dfTemp = dfTemp.add_suffix('_GBP', axis=1)
    df = pd.merge(dfTemp, df, on='Time', how='inner')

    dfTemp = getDataPandas('EURCHF_D1')
    dic['EURCHF'] = normalize(dfTemp)

    dfTemp = dfTemp.add_suffix('_CHF', axis=1)
    df = pd.merge(dfTemp, df, on='Time', how='inner')

    df["WeekDay"] = df.index.weekday
    df = pd.get_dummies(df, columns=['WeekDay'], dtype=int)

    saveData(df, dic)
    print(df.head())
    return df


def newModel(shape):
    with tf.device('/GPU:0'):
        model = Sequential()
        # model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=shape, activation='relu')))
        # model.add(Dropout(0.1))
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=shape, kernel_initializer='glorot_uniform'))
        # model.add(LSTM(64, return_sequences=True, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(LSTM(64, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
        # model.add(LSTM(64, input_shape=(xTrain.shape[1:]), return_sequences=True, kernel_regularizer=regularizers.l2(0.04)))
        model.add(Dropout(0.1))
        model.add(LSTM(32, return_sequences=True, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Flatten())
        # model.add(LSTM(32, input_shape=(xTrain.shape[1:]), return_sequences=True))
        # model.add(LSTM(64, return_sequences=True,activation='tanh'))
        model.add(Dropout(0.1))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam()
    lr_scheduler = LearningRateScheduler(scheduleLearningRate)

    model.build(input_shape=(None,) + shape)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # , Precision(), Recall()
    return model


def getModel(dataShape, c):
    options = ['l', 'c', 'q']
    model: Model
    print(f"{options[0]} - loadModel\n"
          f"{options[1]} - createNewAndTrain\n"
          f"{options[2]} - guit\n")
    x = getInput(options, c)
    if x == options[0]:
        model = loadModel('l')
    elif x == options[1]:
        model = newModel(dataShape)
    elif x == options[2]:
        exit(0)
    print(model.summary())
    return model


defaultOptionsNew = {
    'data': 'n',
    'model': 'c',
    'train': 'z',
    'save': None
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
    'train': 'z',
    'save': None
}
defaultOptionsNone = {
    'data': None,
    'model': None,
    'train': None,
    'save': None
}


def calculatePercentageAcc(pred, rel, thre):
    pred = pred.reshape(-1)
    predBool = (pred >= thre).astype(int)
    accTrain = (predBool == rel).astype(int)
    count_ones = np.count_nonzero(accTrain == 1)
    acc = (count_ones / len(accTrain) * 100)
    return acc

def calculatePercentageClassification(pred,rel,threshold):

    predBool = (pred.reshape(-1) >= threshold).astype(int)
    length = len(predBool)
    print(f'Tabela wyników predykcji dla zbioru testowego (threshold: {threshold}):')
    # Prawdziwie Pozytywny
    PP = np.count_nonzero(np.logical_and((predBool == 1), (rel == 1)).astype(int)) / length * 100
    print(f'Prawdziwie pozytywny: {PP}')

    # Prawdziwie Negatywny
    PN = np.count_nonzero(np.logical_and((predBool == 0), (rel == 0)).astype(int)) / length * 100
    print(f'Prawdziwie negatywny: {PN}')

    # Fałszywie Pozytywny
    FP = np.count_nonzero(np.logical_and((predBool == 1), (rel == 0)).astype(int)) / length * 100
    print(f'Fałszywie pozytywny: {FP}')

    # Fałszywie Newgatywny
    FN = np.count_nonzero(np.logical_and((predBool == 0), (rel == 1)).astype(int)) / length * 100
    print(f'Fałszywie negatywny: {FN}')

def main():

    # # Ustaw backend Keras na TensorFlow
    # k.backend.set_image_data_format('channels_last')
    #
    # Skonfiguruj dostęp do GPU
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.Session(config=config)
    # k.set_session(session)


    defaultOptions = defaultOptionsLoadOnlyData
    df = getData(defaultOptions['data'])
    timeSeriesTuple = getTimeSeriesDataBool(df.to_numpy(), df.columns.get_loc('Close_USD'), 30, 5, 1)
    trainData, testData = splitData(timeSeriesTuple, 0.1)
    dataShape = trainData[0].shape[1:]
    print(f'\nData shape: {dataShape}\n')
    model = getModel(dataShape, defaultOptions['model'])

    # Trenowanie modelu
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

    trainPrediction = model.predict(trainData[0])
    testPrediction = model.predict(testData[0])
    print(testPrediction.reshape(-1)[:20])
    print((testPrediction.reshape(-1) >= 0.5).astype(int))

    threshold = 0.50
    print(f'threshold: {threshold}')
    print(f'Celnosc na zbiorze uczącym = {calculatePercentageAcc(trainPrediction, trainData[1], threshold)}')
    print(f'Celnosc na zbiorze testowym/validacyjnym = {calculatePercentageAcc(testPrediction, testData[1], threshold)}')

    calculatePercentageClassification(testPrediction,testData[1],threshold)

    threshold = 0.52
    print(f'\nthreshold: {threshold}')
    print(f'Celnosc na zbiorze uczącym = {calculatePercentageAcc(trainPrediction, trainData[1], threshold)}')
    print(f'Celnosc na zbiorze testowym/validacyjnym = {calculatePercentageAcc(testPrediction, testData[1], threshold)}')

    calculatePercentageClassification(testPrediction,testData[1],threshold)

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
