import csv
import random
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
    classes_regression = None
    classes_classification = None

    # The class labels
    class_labels = None
    # The numeric class labels
    class_labels_regression = None
    class_labels_classification = None
    # The normalized feature values
    feature_numeric = None

    class_weight = None

    rank_steps = [
        0.08021739130434782,
        0.10652173913043478,
        0.13260869565217392,
        0.1591304347826087,
        0.18478260869565216,
        0.21130434782608695,
        0.23782608695652174,
        0.26456521739130434,
        0.29891304347826086,
        0.3332608695652174,
        0.36782608695652175,
        0.4032608695652174,
        0.44673913043478264,
        0.4902173913043478,
        0.5334782608695652,
        0.5769565217391304,
        0.6202173913043478,
        0.6641304347826087
    ]

    avg_mmr = [
        0.0391304347826087,
        0.0941304347826087,
        0.11869565217391304,
        0.14630434782608695,
        0.17195652173913045,
        0.19782608695652174,
        0.22478260869565217,
        0.25086956521739134,
        0.28108695652173915,
        0.31630434782608696,
        0.3506521739130435,
        0.38521739130434784,
        0.425,
        0.46847826086956523,
        0.5119565217391304,
        0.5552173913043479,
        0.5986956521739131,
        0.6419565217391304,
        0.8336956521739131,
    ]

    def classify(self, mmr):
        c = 0
        for sep in self.rank_steps:
            if mmr > self.rank_steps[c]:
                c += 1
        return c

class DataPrep:
    def prep_data(self, fname):
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

        # Using equidistant values 0-1
        #d.classes_regression = np.array([i for i in range(len(d.classes))])
        # Using scaled mmr values
        d.classes_regression = d.avg_mmr
        d.classes_classification = np.array([i for i in range(len(d.classes))], dtype=np.int64)

        # Grab labels
        d.class_labels = np.array([i[0] for i in f[1:]])
        indices = np.array([d.classes.index(i) for i in d.class_labels], dtype=np.int64)
        d.class_labels_classification = np.array([d.classes_classification[i] for i in indices])
        d.class_labels_regression = np.array([d.classes_regression[i] for i in indices])

        # Grab and normalize features
        feature_values = np.array([i[1:] for i in f[1:]], dtype=np.float64)
        d.feature_numeric = stats.zscore(feature_values, axis=0)


        d.class_weight = class_weight.compute_class_weight(
            'balanced',
            d.classes,
            d.class_labels
        )
        d.class_weight = {i:d.class_weight[i] for i in range(len(d.classes))}

        return d
