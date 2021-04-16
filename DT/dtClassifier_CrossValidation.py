import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import plot_confusion_matrix


def main():
    # read data
	df = pd.read_csv('data_trimmed.csv')
	df['rank'] = df['rank'].astype('category').cat.reorder_categories(['bronze 1','bronze 2','bronze 3','silver 1','silver 2','silver 3','gold 1','gold 2','gold 3','platinum 1','platinum 2','platinum 3','diamond 1','diamond 2','diamond 3','champion 1','champion 2','champion 3','grand champion'])
	ranks = ['B1','B2','B3','S1','S2','S3','G1','G2','G3','P1','P2','P3','D1','D2','D3','C1','C2','C3','GC']
	df.sort_values(by=["rank"],inplace=True)
	df.reset_index(drop=True,inplace=True)

    # shuffle data
	df_shuffled = df.sample(frac=1).reset_index(drop=True)
	X = df_shuffled.iloc[:,1:].to_numpy()
	y = df_shuffled.iloc[:,0].to_numpy()

    # split data for cross validation
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

	# create decision tree object
	dTree = DecisionTreeClassifier(criterion='entropy')

	# GridSearchCV with train data
	dTree = gridSearch(X_train,y_train, dTree)

	# get accuracy on test data
	start = time.time()
	dTree.fit(X_train, y_train)
	print("Training took {} seconds".format(time.time()-start))
	print("Accuracy:", dTree.score(X_test,y_test))

    # plot diagrams
	plt.rcParams.update({'font.size': 5})
	plot_confusion_matrix(dTree, X_test, y_test,labels=pd.unique(df['rank']).tolist(),display_labels=ranks,normalize='true')
	plt.savefig('dTree_confusion_matrix.png')
	
	plot_tree(dTree, feature_names=df.columns[1:], impurity=False, label='none', max_depth=3, filled=True) 
	plt.savefig('dTree_decision_tree.png')

def gridSearch(X, y, dTree):
    print('gridSearch')

    # pick paramters to iterate over
    splitter		    = ['best']
    max_depth		    = [9, 10, 15, 20, 25]
    min_samples_split	= [500, 1000, 1500, 2000, 2500]
    min_samples_leaf	= [200, 500, 1000, 1500, 2000, 2500]
    max_features		= [40, 50, None]
	
    param_grid = {'splitter':splitter, 'max_depth': max_depth,
		          'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
		          'max_features': max_features}

	# start cross validation search search
    print('running search')
    start = time.time()
    grid_search = GridSearchCV(dTree, param_grid, cv=10, scoring='accuracy', n_jobs=1, verbose=4)
    grid_search.fit(X, y)
    print("Grid search took {} seconds".format(time.time()-start))

    # print best parameters it found and return dTree
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search.best_estimator_


if __name__ == '__main__':
	main()












