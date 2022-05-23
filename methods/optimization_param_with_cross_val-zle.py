import numpy as np
import strlearn as sl
import autograd.numpy as anp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.base import clone
# from pymoo.model.problem import Problem
from pymoo.core.problem import ElementwiseProblem


class OptimizationParamCrossVal(ElementwiseProblem):
    def __init__(self, X, y, test_size, estimator, scale_features, n_features, n_param=2, objectives=2, random_state=0, feature_names=None):

        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_param = n_param
        self.scale_features = scale_features
        self.n_features = n_features

        self.X = X
        self.y = y
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, stratify=self.y)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(self.X), np.copy(self.y), np.copy(self.X), np.copy(self.y)

        self.folds = []
        # Repeated Stratified K-Fold cross validator
        # n_splits = 2
        # n_repeats = 5
        # n_folds = n_splits * n_repeats
        # for train_index, test_index in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234).split(X, y):
        #     self.folds.append(((X[train_index], y[train_index]), (X[test_index], y[test_index])))

        for train_index, test_index in KFold(n_splits=10).split(X, y):
            self.folds.append(((X[train_index], y[train_index]), (X[test_index], y[test_index])))

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        xl_real = [1E6, 1E-7]
        xl_binary = [0] * n_features
        xl = np.hstack([xl_real, xl_binary])
        xu_real = [1E9, 1E-4]
        xu_binary = [1] * n_features
        xu = np.hstack([xu_real, xu_binary])
        n_variable = self.n_param + self.n_features

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=1, xl=xl, xu=xu)

    # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
    def validation(self, x):
        C = x[0]
        gamma = x[1]
        selected_features = x[2:]
        selected_features = selected_features.tolist()

        for (X_train, y_train), (X_test, y_test) in self.folds:
            # If at least one element in x are True
            if True in selected_features:
                clf = None
                clf = clone(self.estimator.set_params(C=C, gamma=gamma))
                clf.fit(X_train[:, selected_features], y_train)
                y_pred = clf.predict(X_test[:, selected_features])
                metrics = [sl.metrics.precision(y_test, y_pred), sl.metrics.recall(y_test, y_pred)]
            else:
                metrics = [0, 0]
            return metrics

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # print(scores)

        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores[0]
        f2 = -1 * scores[1]
        out["F"] = anp.column_stack(np.array([f1, f2]))

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_features)
        out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
