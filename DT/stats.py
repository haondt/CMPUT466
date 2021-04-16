import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import seaborn as sns
import numpy as np


def main():
    # read data
    df = pd.read_csv('data_ordered.csv')
    df['rank'] = df['rank'].astype('category').cat.reorder_categories(['B1','B2','B3','S1','S2','S3','G1','G2','G3','P1','P2','P3','D1','D2','D3','C1','C2','C3','GC'])
    ranks = ['B1','B2','B3','S1','S2','S3','G1','G2','G3','P1','P2','P3','D1','D2','D3','C1','C2','C3','GC']
    df.sort_values(by=["rank"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # shuffle data
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    X = df_shuffled.iloc[:,1:].to_numpy()
    y = df_shuffled.iloc[:,0].to_numpy()

    # split data for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
	# create decision tree object
    
    dTree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=9)
    
    # train tree
    start = time.time()
    dTree.fit(X_train, y_train)
    trueScore = dTree.score(X_test,y_test)
    print("Accuracy: {}".format(trueScore*100))
    
    # permutation test
    print("shuffled y")
    scores = []
    above = []
    for i in range(1000):
        yTest = y_test
        np.random.shuffle(yTest)
        score = dTree.score(X_test,yTest)
        scores.append(score)
        #print("Accuracy: {}".format(score*100))
        if (score >= trueScore):
            above.append(score)
    
    print("random average: {}".format(sum(scores)/len(scores)*100))
    print("above:", len(above))
    print("total:", len(scores))    
    print(len(above) / len(scores))
    print("Stats took {} seconds".format(time.time()-start))
    
    
if __name__ == '__main__':
	main()

