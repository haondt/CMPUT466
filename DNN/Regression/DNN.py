
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.layers import ReLU, Dense, Flatten, Dropout
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

class CVTuner(Tuner):
    def run_trial(self, trial, *fit_args, **fit_kwargs):
        cv = model_selection.KFold(5)
        val_losses = []
        #batch_size = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
        batch_size = 50000
        #epochs = trial.hyperparameters.Int('epochs', 10, 30)
        x = fit_kwargs['x']
        y = fit_kwargs['y']
        epochs=1
        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, class_weight=fit_kwargs['class_weight'])
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {
            'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)





def create_regression_model():
    model = Sequential()
    model.add(Dense(74, activation="sigmoid", input_shape=(65,), activity_regularizer=None))
    #model.add(Dropout(0.66186))

    model.add(Dense(170, activation="relu"))
    #model.add(Dropout(0.19215))
    model.add(Dense(30, activation="tanh"))
    #model.add(Dropout(0.52789))
    model.add(Dense(70, activation="sigmoid"))
    #model.add(Dropout(0.46985))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer='adamax',
        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_regression_model_mmr():
    model = Sequential()
    model.add(Dense(
        72,
        activation="sigmoid",
        input_shape=(65,),
        activity_regularizer=None
    ))

    model.add(Dense(110, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(60, activation="tanh"))
    model.add(Dense(130, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer='adamax',
        loss=r2, metrics=['accuracy'])
    return model

def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return -( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_regression_model_cv():
    model = Sequential()
    model.add(Dense(74, activation="sigmoid", input_shape=(65,), activity_regularizer=None))
    model.add(Dropout(0.14307))

    model.add(Dense(20, activation="sigmoid"))
    model.add(Dropout(0.78059))
    model.add(Dense(140, activation="relu"))
    model.add(Dropout(0.5164))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adamax(
            learning_rate=1.94e-05,
            beta_1=0.9456,
            beta_2=0.91756
        ),
        loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    return model


def regression_model_builder_X(hp):
    model = Sequential()
    none = lambda x: x if x is not 'none' else None
    model.add(
        Dense(
            hp.Int('units_first_layer',30,100,step=10, default=70),
            input_shape=(65,),
            activation=hp.Choice('activation_first_layer', values=['relu','sigmoid','tanh', 'linear']),
            activity_regularizer=none(hp.Choice('a_reg_first_layer', values=['l1', 'l2', 'none'])),
            kernel_regularizer=none(hp.Choice('k_reg_first_layer', values=['l1', 'l2','none']))
        )
    )

    for i in range(hp.Int('additional_layers', 2, 30, default=5)):
        if hp.Choice('layer_type_' + str(i), values=['dense', 'dropout']) == 'dense':
            model.add(Dense(
                units=hp.Int('units_' + str(i), 10, 100, step=10),
                activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid', 'tanh', 'linear']),
                activity_regularizer=none(hp.Choice('a_reg_' + str(i), values=['l1', 'l2', 'none'])),
                kernel_regularizer=none(hp.Choice('k_reg_' + str(i), values=['l1', 'l2', 'none']))
            ))
        else:
            model.add(Dropout(hp.Float('dropout_' + str(i) , min_value=0,max_value=1,default=0.4 )))

    # Coalescing layer for single regression value
    model.add(Dense(1, activation=hp.Choice('activation_last_layer', values=['linear', 'sigmoid', 'tanh', 'relu'])))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','sgd','adagrad', 'adamax']),
        loss='mean_squared_logarithmic_error',
        metrics=['accuracy']
    )
    return model
def regression_model_builder(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_first_layer',30,100,step=10), input_shape=(65,),
        activation=hp.Choice('activation_first_layer', values=['relu','sigmoid','tanh'])))

    for i in range(hp.Int('additional_layers', 2, 50)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), 10, 100, step=10),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid', 'tanh'])
        ))

    # Coalescing layer for single regression value
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','sgd','adagrad', 'adamax']),
        loss=hp.Choice('loss_function', values=['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error']),
        metrics=['accuracy']
    )
    return model

def regression_model_builder_2(hp):
    # Refine tuning
    '''
    Results:
    {
        'units_first_layer': 74,
        'activation_first_layer': 'sigmoid',
        'additional_layers': 3,
        'units_0': 170,
        'activation_0': 'relu',
        'units_1': 30,
        'activation_1': 'tanh',
        'units_2': 70,
        'activation_2': 'sigmoid',
        'optimizer': 'adamax'
    }
    best val_loss = 0.006592826917767525
    '''
    model = Sequential()
    model.add(Dense(hp.Int('units_first_layer',60,80), input_shape=(65,),
        activation=hp.Choice('activation_first_layer', values=['relu','sigmoid','tanh'])))

    for i in range(hp.Int('additional_layers', 2, 5)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), 10, 200, step=10),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid', 'tanh'])
        ))

    # Coalescing layer for single regression value
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam','adagrad', 'adamax']),
        loss=r2,
        metrics=['accuracy']
    )
    return model

def regression_model_builder_3(hp):
    # add dropout layers and tune optimizer
    '''
    '''
    model = Sequential()
    model.add(Dense(
        74,
        input_shape=(65,),
        activation='sigmoid')
    )

    model.add(Dropout(hp.Float('dropout_first', min_value=0.01, max_value=0.99, default=0.4)))
    for i in range(hp.Int('additional_layers', 2, 8)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), 10,200,step=10,default=80),
            activation=hp.Choice('activation_' + str(i), values=['relu', 'sigmoid','tanh'])
        ))
        model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.01, max_value=0.99, default=0.4)))

    # Coalescing layer for single regression value
    model.add(Dense(1, activation=hp.Choice('final_layer_activation', values=['relu','sigmoid','tanh', 'linear'])))

    model.compile(
        optimizer=Adamax(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-3),
            beta_1=hp.Float('beta_1', min_value=0.9, max_value=0.99999, sampling='LOG', default=0.9),
            beta_2=hp.Float('beta_2', min_value=0.9, max_value=0.99999, sampling='LOG', default=0.999)
        ),
        loss='mean_squared_logarithmic_error',
        metrics=['accuracy']
    )
    return model




def tune_regression(data):
    tuner = kt.Hyperband(
        regression_model_builder_2,
        objective='val_loss',
        max_epochs=100,
        factor=3,
        directory='.',
        project_name='kt_project_reg_102',
        seed=0
    )
    '''
    tuner = CVTuner(
        hypermodel=regression_model_builder_3,
        oracle=kt.oracles.BayesianOptimization(
            objective='val_loss',
            seed=0,
            max_trials=1000
        )
    )
    '''
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tuner.search(x=data.feature_numeric, y=data.class_labels_regression, validation_split=0.2, callbacks=[stop_early], class_weight=data.class_weight)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

def _test_regression_model(data):
    sample = random.sample(range(len(data.class_labels)), len(data.class_labels)//10)
    mask = np.ones(len(data.class_labels), np.bool)
    mask[sample] = 0
    train_x = data.feature_numeric[mask,:]
    train_y = data.class_labels_regression[mask]
    test_x = data.feature_numeric[~mask,:]
    test_y = data.class_labels_regression[~mask]

    model = create_regression_model()
    hist = model.fit(
        x=train_x,
        y=train_y,
        batch_size=len(train_y),
        epochs=500,
        verbose=1,
        validation_split=0.2,
        class_weight=data.class_weight
    )

    test_y_hat = model.predict(test_x)

    loss, acc = model.evaluate(test_x, test_y, verbose=0)
    print('loss: {}, accuracy: {}'.format(loss, acc))

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    a = plt.axes(aspect='equal')
    plt.scatter(test_y, test_y_hat)
    plt.xlabel('True Values [rank]')
    plt.ylabel('Predictions [rank]')
    lims = [0, 1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def test_regression_model(data):

    kfold = KFold(n_splits=5, shuffle=True)
    accs = []
    loss = []
    for train, test in kfold.split(data.feature_numeric, data.class_labels_regression):
        model = create_regression_model_mmr()
        model.fit(
            x=data.feature_numeric[train,:],
            y=data.class_labels_regression[train],
            batch_size=50000,
            epochs=300,
            verbose=1
        )

        # calculate regression loss
        scores = model.evaluate(data.feature_numeric[test], data.class_labels_regression[test], verbose=0)
        loss.append(scores[0])

        # calculate classification accuracy
        #acc.append(scores[1]*100)
        y_test = data.class_labels_classification[test]
        y_hat_test = np.array([data.classify(i) for i in model.predict(data.feature_numeric[test], verbose=0)], dtype=np.int64)

        acc = len(np.where(y_test == y_hat_test))/len(y_test)
        accs.append(acc)

    print("accs:", accs)
    print("losses:", loss)
    print("avg acc", np.mean(accs))
    print("avg loss", np.mean(loss))

def plot_regression_model(data):
    sample = random.sample(range(len(data.class_labels)), len(data.class_labels)//10)
    mask = np.ones(len(data.class_labels), np.bool)
    mask[sample] = 0
    train_x = data.feature_numeric[mask,:]
    train_y = data.class_labels_regression[mask]
    test_x = data.feature_numeric[~mask,:]
    test_y = data.class_labels_regression[~mask]

    model = create_regression_model_mmr()
    hist = model.fit(
        x=train_x,
        y=train_y,
        batch_size=len(train_y),
        epochs=400,
        verbose=1,
        validation_split=0.2
    )


    test_y_hat = model.predict(test_x)

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    a = plt.axes(aspect='equal')
    plt.scatter(test_y, test_y_hat)
    plt.xlabel('True Values [rank]')
    plt.ylabel('Predictions [rank]')
    lims = [-0.1, 1.1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    a = plt.axes(aspect='equal')
    plt.scatter(test_y_hat, test_y)
    plt.ylabel('True Values [rank]')
    plt.xlabel('Predictions [rank]')
    lims = [-0.1, 1.1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


    binned_pred = np.round(test_y_hat, 1)
    preds = [i/10 for i in range(10)]
    real_y_sums = [[] for i in range(len(preds))]
    for i in range(len(test_y)):
        ind = preds.index(binned_pred[i])
        real_y_sums[ind].append(test_y[i])

    avg_real_ys = np.array([np.mean(i) for i in real_y_sums])

    a = plt.axes(aspect='equal')
    plt.plot( avg_real_ys, preds)
    plt.legend(['average true value'])
    plt.ylabel('True Values [rank]')
    plt.xlabel('Binned Predictions [rank]')
    lims = [-0.1, 1.1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    ys = list(sorted(set(test_y)))
    ysums = [[] for i in range(len(ys))]
    for i in range(len(test_y)):
        ind = ys.index(test_y[i])
        ysums[ind].append(test_y_hat[i])

    avg_preds = np.array([np.mean(i) for i in ysums])

    a = plt.axes(aspect='equal')
    plt.plot(ys, avg_preds)
    plt.legend(['average prediction'])
    plt.xlabel('True Values [rank]')
    plt.ylabel('Predictions [rank]')
    lims = [-0.1, 1.1]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def main():
    # Data file
    fname = 'data_trimmed.csv'

    # Make numpy arrays print nicer
    np.set_printoptions(suppress=True)

    # Prepare data
    # TODO: Move normalization from prep_data to layer within DNN
    data = prep_data(fname)


    # Prevent full gpu memory allocation
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    tune_regression(data)
    #tune_classification(data)

    #test_regression_model(data)
    #test_classification_model(data)

    #plot_regression_model(data)



if __name__ == '__main__':
    main()


