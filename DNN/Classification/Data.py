import csv
import random
import json
import numpy as np
from sklearn.utils import class_weight
from scipy import stats


class Data:
    # The names (titles) of each feature
    feature_names = None
    # The name (title) of the classification
    class_name = None

    # The names of the possible classifications
    classes = None
    # The numeric representation of the possible classifications
    classes_numeric = None

    # The class labels
    class_labels = None
    # The numeric class labels
    class_labels_numeric = None

    # The normalized feature values
    feature_numeric = None

    # The weights for each class
    class_weight = None

    # Serialize self into a json string
    def jsonify(self):
        d = {
            'feature_names': self.feature_names,
            'class_name': self.class_name,
            'classes': self.classes,
            'classes_numeric': self.classes_numeric.tolist(),
            'class_labels': self.class_labels.tolist(),
            'class_labels_numeric': self.class_labels_numeric.tolist(),
            'feature_numeric': self.feature_numeric.tolist(),
            'class_weight': self.class_weight
        }
        return json.dumps(d, indent=4, sort_keys=True)

    # Deserialize json dictionary into self
    def load_json(self, s):
        self.feature_names = s['feature_names']
        self.class_name = s['class_name']
        self.classes = s['classes']
        self.classes_numeric = np.array([int(i) for i in s['classes_numeric']])
        self.class_labels = np.array(s['class_labels'])
        self.class_labels_numeric = np.array(s['class_labels_numeric'])
        self.feature_numeric = np.array([np.array(i) for i in s['feature_numeric']])
        self.class_weight = {int(i):s['class_weight'][i] for i in s['class_weight']}



class DataPrep:

    @staticmethod
    def prep_data(fname):
        # Read csv
        f = [i for i in csv.reader(open(fname, 'r'), delimiter=',')]

        # shuffle data
        headers = f[0]
        order = [i+1 for i in range(len(f[1:]))]
        random.shuffle(order)
        f = [headers] + [f[i] for i in order]

        # Grab feature names from data
        d = Data()
        d.feature_names = f[0][1:]
        d.class_name = f[0][0]

        # Grab label range
        d.classes = [[i + ' ' + j for j in '123'] for i in
            [
                'bronze',
                'silver',
                'gold',
                'platinum',
                'diamond',
                'champion'
            ]
        ]
        d.classes.append(['grand champion'])
        d.classes = [i for s in d.classes for i in s]

        # List classes as integers
        d.classes_numeric = np.array([i for i in range(len(d.classes))], dtype=np.int64)

        # Grab labels
        d.class_labels = np.array([i[0] for i in f[1:]])
        indices = np.array([d.classes.index(i) for i in d.class_labels], dtype=np.int64)
        d.class_labels_numeric = np.array([d.classes_numeric[i] for i in indices])

        # Grab and normalize features
        feature_values = np.array([i[1:] for i in f[1:]], dtype=np.float64)
        d.feature_numeric = stats.zscore(feature_values, axis=0)

        # Generate class weights
        d.class_weight = class_weight.compute_class_weight(
            'balanced',
            d.classes,
            d.class_labels
        )
        d.class_weight = {i:d.class_weight[i] for i in range(len(d.classes))}

        return d

    # Split a dataset  into train / test datasets
    @staticmethod
    def split_data(data):
        train = Data()
        test = Data()

        for i in [train, test]:
            i.feature_names = data.feature_names
            i.class_name = data.class_name
            i.classes = data.classes
            i.classes_numeric = data.classes_numeric
            i.class_weight = data.class_weight


        # 20% for test
        sample = random.sample(range(len(data.class_labels)), int(len(data.class_labels)*0.2))
        mask = np.ones(len(data.class_labels), np.bool)
        mask[sample] = 0

        train.class_labels = data.class_labels[mask]
        train.class_labels_numeric = data.class_labels_numeric[mask]
        train.feature_numeric = data.feature_numeric[mask,:]

        test.class_labels = data.class_labels[~mask]
        test.class_labels_numeric = data.class_labels_numeric[~mask]
        test.feature_numeric = data.feature_numeric[~mask,:]

        return train, test


    # Save a dataset to file
    @staticmethod
    def save_data(data, fname):
        with open(fname, 'w') as f:
            f.write(data.jsonify())

    # Load a dataset from file
    @staticmethod
    def load_data(fname):
        d = Data()
        s = None
        with open(fname, 'r') as f:
            s = json.load(f)
        d.load_json(s)
        return d