####################
# ML Hwk 2
# Author: Kevin Thompson
# Last Updated: June 6, 2020
###################

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from scipy.stats import rv_continuous, uniform
from copy import copy


class logUniform_generator(rv_continuous):
    def _cdf(self, x):
        return np.log(x/self.a)/np.log(self.b/self.a)

def logUniform(a=1, b=np.exp(1)):
    return logUniform_generator(a=a, b=b, name='logUniform')


warnings.filterwarnings('ignore')

X = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
y = np.ones(X.shape[0])
y[2:5] = 0
n_folds = 5

rf_params = {'clf': RandomForestClassifier(),
             'n_estimators': [ 10, 20],
             'criterion': ['gini', 'entropy'],
             'min_samples_leaf': [1]}
lr_params = {'clf': LogisticRegression(),
             'C': np.array(logUniform().rvs(size=100)),
             'penalty': ['l2', 'none'],
             'solver': ['newton-cg', 'lbfgs']}
params = [lr_params, rf_params]


def grid_search(X, y, params_list, n_folds, scoring):
    cv = KFold(n_splits=n_folds)
    ret = list()

    for params in params_list:
        clf = params.pop('clf')
        grid = ParameterGrid(params)

        for element in grid:
            for train_indices, test_indices in cv.split(X,y):
                clf = clf.set_params(**element)
                clf.fit(X[train_indices], y[train_indices])
                pred = clf.predict(X[test_indices])
                clf_info  = {'train_indices': train_indices,
                             'test_indices': test_indices,
                             'score': scoring(y[test_indices], pred),
                             'params': clf.get_params()}
                ret.append((clf.__class__.__name__,  clf_info))
    return ret

