import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def v1():
    print(tf.__version__)

    url = 'iris.data'
    column_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'class'
    ]

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=',', skipinitialspace=True)

    dataset = raw_dataset.copy()

    dataset = dataset.dropna()

    #dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    print(dataset.tail())
    #sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    sns.pairplot(train_dataset, diag_kind='kde', hue='class')
    plt.show()
    exit()

    #print(dataset.tail())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    dnn_model = build_and_compile_model(normalizer)
    #print(dnn_model.summary())

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0,
        epochs=100
    )

    results = dnn_model.evaluate(test_features, test_labels, verbose=1)
    print("MAE: {}".format(results))
    test_predictions = dnn_model.predict(test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def v2():
    print(tf.__version__)

    url = 'data_trimmed.csv'

    column_names = [i for i in csv.reader(open(url, 'r'), delimiter=',')][0]

    raw_dataset = pd.read_csv(url, names=column_names, header=1,
        na_values='', comment='\t',
        sep=',', skipinitialspace=True)

    dataset = raw_dataset.copy()
    ranks = [[i + ' ' + j for j in '123'] for i in
        [
            'bronze',
            'silver',
            'gold',
            'platinum',
            'diamond',
            'champion'
        ]
    ]
    ranks.append(['grand champion'])
    # flatten list
    ranks = [i for s in ranks for i in s]
    rank_mmr = [

    ]

    #dataset['rank'] = np.array([ranks.index(r) for r in dataset['rank']])

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print('plotting...')
    sns.pairplot(data=test_dataset, y_vars=['rank'], x_vars=random.sample(column_names[1:],8), diag_kind='kde')
    plt.show()
    exit()
    #sns.pairplot(test_dataset[['rank'] + random.sample(column_names[1:], 15)], diag_kind='hist', hue='rank')
    sns.pairplot(test_dataset[[
        'rank',
        'bpm',
        'amount collected',
        'percentage supersonic speed'
    ]], diag_kind='kde', hue='rank')
    #sns.pairplot(data=test_dataset, x_vars=['rank'], y_vars=random.sample(column_names[1:],5), diag_kind='kde', hue='rank')
    #plt.show()
    plt.savefig('foo4.png')

    exit()

    #sns.pairplot(train_dataset[['rank'] + random.sample(column_names, 4)], diag_kind='kde')
    #plt.show()

    exit()

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('rank')
    test_labels = test_features.pop('rank')

    #print(train_dataset.describe().transpose()[['mean', 'std']])

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    dnn_model = build_and_compile_model(normalizer)

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=1,
        epochs=100
    )

    results = dnn_model.evaluate(test_features, test_labels, verbose=1)
    print("MAE: {}".format(results))
    test_predictions = dnn_model.predict(test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [rank]')
    plt.ylabel('Predictions [rank]')
    lims = [-1, 20]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def trial(trialData):
    start = time.time()
    #print("Started training " + trialData.name)
    model = train(trialData.train_x, trialData.train_y, trialData.epochs, trialData.batch_size, trialData.class_weight)
    #print("Done training " + trialData.name)
    t = time.time()-start

    mini_y = trialData.test_y[:10]
    mini_y_hat = model.predict_classes(trialData.test_x[:10,:])
    print(mini_y)
    print(mini_y_hat)
    #print("mini acc: {}".format(sum(mini_y == mini_y_hat)/10))
    maxi_y = trialData.test_y
    #maxi_y_hat = model.predict_classes(trialData.test_x)
    maxi_y_hat = model.predict(trialData.test_x)
    print(len(maxi_y))
    print(len(maxi_y_hat))
    print(maxi_y)
    print(maxi_y_hat)

    #print("maxi acc: {}".format(sum(maxi_y == maxi_y_hat)/len(maxi_y)))

    a = plt.axes(aspect='equal')
    plt.scatter(maxi_y, maxi_y_hat)
    plt.xlabel('True Values [rank]')
    plt.ylabel('Predictions [rank]')
    lims = [0, 1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    loss, acc = model.evaluate(trialData.test_x, trialData.test_y, verbose=0)
    #print(trialData.name + " test accuracy: %.3f, batch size: %d" % (acc, trialData.batch_size))
    print("Trial, Accuracy, time, batch size, epochs")
    print("%s, %.3f, %.3f, %d, %d\n" % (trialData.name, acc, t, trialData.batch_size, trialData.epochs), end='')
    return

class TrialData:
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    epochs = None
    batch_size = None
    name = None
    class_weight = None
def run_trial(data):
    sample = random.sample(range(len(data.class_labels)), len(data.class_labels)//10)
    mask = np.ones(len(data.class_labels), np.bool)
    mask[sample] = 0
    train_x = data.feature_numeric[mask,:]
    train_y = data.class_labels_regression[mask]
    test_x = data.feature_numeric[~mask,:]
    test_y = data.class_labels_regression[~mask]
    j = 0
    for epoch in [10000]:
        trialDatum = []
        for bs in [60000]:
            j+=1
            trialData = TrialData()
            trialData.train_x = train_x
            trialData.train_y = train_y
            trialData.test_x = test_x
            trialData.test_y = test_y
            trialData.epochs = epoch
            trialData.class_weight = data.class_weight
            trialData.batch_size = bs
            trialData.name = "Trial " + str(j)
            trialDatum.append(trialData)
        pool = ThreadPool(len(trialDatum))
        pool.map(trial, trialDatum)

def train(X,y, epochs=2000, batch_size=100000, cw=None):
    model = create_model()
    model.fit(X,y, epochs=epochs, batch_size=batch_size, verbose=1, class_weight=cw)
    #model.fit(X,y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def test(model, X, y):
    loss, acc = model.evaluate(X, y, verbose=0)
    print('Test Accuracy: %.3f' % acc)

v2()
