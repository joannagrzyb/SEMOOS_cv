import numpy as np
import strlearn as sl
import autograd.numpy as anp
from sklearn.base import clone
from pymoo.core.problem import ElementwiseProblem


class OptimizationParamCrossVal(ElementwiseProblem):
    def __init__(self, X, y, estimator, scale_features, n_features, cross_validation, n_param=2, objectives=2):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.scale_features = scale_features
        self.n_features = n_features
        self.cross_validation = cross_validation
        self.n_param = n_param
        self.objectives = objectives

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
        scores = np.zeros((self.cross_validation.get_n_splits(), self.objectives))

        for fold_id, (train, test) in enumerate(self.cross_validation.split(self.X, self.y)):
            if True in selected_features:
                X_train = self.X[train]
                y_train = self.y[train]
                X_test = self.X[test]
                y_test = self.y[test]
                clf = None
                clf = clone(self.estimator.set_params(C=C, gamma=gamma))
                clf.fit(X_train[:, selected_features], y_train)
                y_pred = clf.predict(X_test[:, selected_features])

                scores[fold_id] = [sl.metrics.precision(y_test, y_pred), sl.metrics.recall(y_test, y_pred)]
            else:
                scores = [0, 0]

        return np.mean(scores, axis=0)

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)

        # It is needed to prevent from IndexError
        if not isinstance(scores, (np.ndarray)):
            scores = [0, 0]

        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores[0]
        f2 = -1 * scores[1]
        out["F"] = anp.column_stack(np.array([f1, f2]))

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_features)
        out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2
