
from tensorflow.keras import backend as K
from scipy.stats import gaussian_kde
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.layers import ReLU, Dense, Flatten, Dropout
from tensorflow.keras.models import model_from_json
from matplotlib.colors import LogNorm

import random
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import kerastuner as kt
from tensorflow.python.keras.backend import set_session
from matplotlib import pyplot as plt
from sklearn import model_selection
from kerastuner.engine.tuner import Tuner

from Data import Data, DataPrep

from lime import lime_tabular


first_layer_size=65

class cv_tuner(Tuner):
    class_weight = None
    def run_trial(self, trial, x,y, batch_size, epochs):
        cv = model_selection.KFold(5)
        val_loss = []
        val_acc = []

        for train_indices, test_indices in cv.split(x):
            x_train = x[train_indices]
            y_train = y[train_indices]
            x_test = x[test_indices]
            y_test = y[test_indices]

            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=self.class_weight)
            loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
            #val_loss.append(model.evaluate(x_test, y_test))
            val_loss.append(loss)
            val_acc.append(accuracy)

        self.oracle.update_trial(trial.trial_id, {
            'val_loss': np.mean(val_loss),
            'val_acc': np.mean(val_acc)
        })
        self.save_model(trial.trial_id, model)

def create_classification_model():
    model = Sequential()
    model.add(Dense(39, input_shape=(first_layer_size,), activation='tanh'))
    model.add(Dense(
        units=28,
        activation='relu'
    ))
    model.add(Dropout(0.098954))
    model.add(Dense(
        units=32,
        activation='sigmoid'
    ))
    model.add(Dropout(0.027818))
    model.add(Dense(19, activation="softmax"))

    model.compile(
        optimizer=Adam(
            learning_rate=0.0025923,
            beta_1=0.90489,
            beta_2=0.99883
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def classification_model_builder(hp):
    '''
    Results:
    Best val_acc So Far: 0.19982486963272095
    Total elapsed time: 01h 36m 30s

    Search: Running Trial #254

    Hyperparameter    |Value             |Best Value So Far
    units_first_layer |50                |40
    activation_firs...|sigmoid           |tanh
    additional_layers |4                 |2
    units_0           |80                |50
    activation_0      |tanh              |tanh
    dropout_0         |0.51309           |0.11214
    units_1           |50                |30
    activation_1      |relu              |tanh
    dropout_1         |0.37365           |0.16723
    final_layer_units |90                |70
    learning_rate     |0.00046468        |0.0025923
    beta_1            |0.91857           |0.90489
    beta_2            |0.97514           |0.99883
    units_2           |30                |80
    activation_2      |relu              |sigmoid
    dropout_2         |0.58616           |0.097376
    units_3           |30                |30
    activation_3      |sigmoid           |sigmoid
    dropout_3         |0.88737           |0.72387
    units_4           |70                |50
    activation_4      |relu              |relu
    dropout_4         |0.42381           |0.31305
    units_5           |50                |70
    activation_5      |relu              |sigmoid
    dropout_5         |0.85436           |0.43852
    units_6           |60                |40
    activation_6      |tanh              |relu
    dropout_6         |0.63697           |0.22963
    units_7           |20                |20
    activation_7      |relu              |tanh
    dropout_7         |0.16532           |0.61968
    tuner/epochs      |100               |100
    tuner/initial_e...|0                 |34
    tuner/bracket     |0                 |2
    tuner/round       |0                 |2

    '''
    model = Sequential()
    model.add(Dense(hp.Int('units_first_layer',40,80,step=10), input_shape=(first_layer_size,),
        activation=hp.Choice('activation_first_layer', values=['relu','sigmoid','tanh'])))

    for i in range(hp.Int('additional_layers', 2, 8)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), 20, 80, step=10),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid', 'tanh'])
        ))
        model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.01, max_value=0.99, default=0.4)))

    model.add(Dense(hp.Int('final_layer_units', 30,100,step=10), activation="softmax"))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-3),
            beta_1=hp.Float('beta_1', min_value=0.9, max_value=0.99999, sampling='LOG', default=0.9),
            beta_2=hp.Float('beta_2', min_value=0.9, max_value=0.99999, sampling='LOG', default=0.999)
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def classification_model_builder_refined(hp):
    '''
    Results:
    Acheived 0.20034334063529968 val_accuracy, settings copied into
    create_classification_model
    '''
    model = Sequential()
    model.add(Dense(hp.Int('units_first_layer',35,45,step=1), input_shape=(first_layer_size,),
        activation='tanh'))

    for i in range(hp.Int('additional_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), 20, 60, step=1),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid', 'tanh'])
        ))
        model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.01, max_value=0.25, default=0.1)))

    model.add(Dense(19, activation="softmax"))

    model.compile(
        optimizer=Adam(
            learning_rate=0.0025923,
            beta_1=0.90489,
            beta_2=0.99883
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_classification(data):
    tuner = cv_tuner(
        hypermodel=classification_model_builder_refined,
        oracle=kt.oracles.Hyperband(
            objective='val_acc',
            max_epochs=100,
            factor=3
        ),
        overwrite=True,
    )

    tuner.class_weight = data.class_weight
    tuner.search(data.feature_numeric, data.class_labels_numeric, batch_size=64, epochs=2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

def train_classification_model(train):

    model = create_classification_model()
    hist = model.fit(
        x=train.feature_numeric,
        y=train.class_labels_numeric,
        batch_size=100000,
        epochs=500,
        verbose=1,
        class_weight=train.class_weight,
        validation_split=0.2
    )

    # save model
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        f.write(model_json)
    model.save_weights('model.h5')

    # plot history
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_model():
    with open('model.json', 'r') as f:
        model_json = f.read()

    model = model_from_json(model_json)
    model.load_weights('model.h5')
    model.compile(
        optimizer=Adam(
            learning_rate=0.0025923,
            beta_1=0.90489,
            beta_2=0.99883
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def test_classification_model(test, model=None):

    if model is None:
        model = get_model()

    scores = model.evaluate(
        x=test.feature_numeric,
        y=test.class_labels_numeric,
        verbose=0,
    )
    acc = scores[1]
    loss = scores[0]

    print('acc:', acc)
    print('loss:', loss)

def plot_classification_model(test, model=None, plot_type='alpha'):
    if model is None:
        model = get_model()

    test_y_hat = model.predict_classes(test.feature_numeric)

    if plot_type == 'alpha' or plot_type == 'heat':
        # plot predictions vs true
        a = plt.axes(aspect='equal')
        lims = [-1,19]
        _ = plt.plot(lims, lims)

        if plot_type == 'alpha':
            # Plot using transparency
            plt.scatter(test.class_labels_numeric, test_y_hat, alpha=0.05)
        else:
            # Plot using colors
            points = []
            counts = {}
            for i in range(len(test_y_hat)):
                x = test.class_labels_numeric[i]
                y = test_y_hat[i]
                xy = (x,y)
                points.append(xy)
                if xy not in counts:
                    counts[xy] = 0
                counts[xy] += 1
            x = [i[0] for i in points]
            y = [i[1] for i in points]
            z = [counts[i]/len(test_y_hat)*100 for i in points]

            plt.scatter(x,y, c=z, cmap=plt.cm.jet, norm=LogNorm())

            cbar = plt.colorbar()
            cbar.set_label('Percentage of total labels [%]', rotation=270, labelpad=15)

        plt.xlabel('True Values [rank]')
        plt.ylabel('Predictions [rank]')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()

    elif plot_type == 'avg_per_rank':
        # Plot average prediction of each rank
        a = plt.axes(aspect='equal')
        preds = {i:[] for i in range(19)}
        for i in range(len(test_y_hat)):
            preds[test.class_labels_numeric[i]].append(test_y_hat[i])
        preds = [np.mean(preds[i]) for i in range(19)]
        plt.plot(list(range(19)), preds)
        plt.legend(['Average Prediction'])

        lims = [-1,19]
        _ = plt.plot(lims, lims)
        plt.xlabel('True Values [rank]')
        plt.ylabel('Prediction [rank]')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()

    elif plot_type == 'box_per_rank':
        # Show distribution of predictions with a box plot for each rank
        a = plt.axes(aspect='equal')
        preds = {i:[] for i in range(19)}
        for i in range(len(test_y_hat)):
            preds[test.class_labels_numeric[i]].append(test_y_hat[i])
        preds = [preds[i] for i in range(19)]
        plt.boxplot(preds, positions=list(range(19)))

        lims = [-1,19]
        _ = plt.plot(lims, lims)
        plt.xlabel('True Values [rank]')
        plt.ylabel('Prediction [rank]')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()

    elif plot_type == 'violin_per_rank':
        # Show distribution of predictions with a violin plot for each rank
        plt.figure(figsize=(20,6), dpi=80)
        preds = {i:[] for i in range(19)}
        for i in range(len(test_y_hat)):
            preds[test.class_labels_numeric[i]].append(test_y_hat[i])
        preds = [preds[i] for i in range(19)]
        plt.violinplot(preds, positions=list(range(19)), showmeans=True)

        lims = [-1,19]
        _ = plt.plot(lims, lims)
        plt.xlabel('True Values [rank]')
        plt.ylabel('Prediction [rank]')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
    elif plot_type == 'std_dev_per_rank':
        # Show distribution of predictions with a box plot for each rank
        preds = {i:[] for i in range(19)}
        for i in range(len(test_y_hat)):
            preds[test.class_labels_numeric[i]].append(test_y_hat[i])
        preds = [preds[i] for i in range(19)]
        std = [np.std(i) for i in preds]
        plt.bar(list(range(19)),std)

        plt.xlabel('Rank')
        plt.ylabel('Standard Deviation')
        plt.show()


def explain_model(test: Data, train: Data, model=None):
    if model is None:
        model = get_model()


    sample = np.array([test.feature_numeric[25]])
    print(model.predict_classes(sample))

    explainer = lime_tabular.LimeTabularExplainer(
        train.feature_numeric,
        feature_names=train.feature_names,
        class_names=train.class_labels_numeric,
        discretize_continuous=True
    )
    exp = explainer.explain_instance(
        test.feature_numeric[25],
        model.predict_proba,
        num_features=first_layer_size,
        top_labels=19
    )

    print(test.class_labels_numeric[25])



    exp.save_to_file('lime.html',show_table=True, show_all=False)




def main():
    # Data file
    fname = 'data/data_trimmed.csv'

    # Make numpy arrays print nicer
    np.set_printoptions(suppress=True)

    # Prepare and cache data
    try:
        train = DataPrep.load_data("train.json")
        test = DataPrep.load_data("test.json")
    except IOError:
        data = DataPrep.prep_data(fname)
        train, test = DataPrep.split_data(data)
        DataPrep.save_data(train, 'train.json')
        DataPrep.save_data(test, 'test.json')

    # Prevent full gpu memory allocation
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Tune hyperparameters
    #tune_classification(train)

    # Train model
    #train_classification_model(train)

    # Test model
    #test_classification_model(test)

    # Test model and plot result
    #for p in ['heat', 'avg_per_rank','box_per_rank','violin_per_rank','std_dev_per_rank']:
        #plot_classification_model(test, plot_type=p)

    explain_model(test,train)



if __name__ == '__main__':
    main()


