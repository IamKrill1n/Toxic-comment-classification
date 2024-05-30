import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class NbLrClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, solver = 'lbfgs', dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.solver = solver
        self.n_jobs = n_jobs

    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def predict(self, x):
        # Verify that model has been fit
        # check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        # check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))
    
    def pre_fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x,1,y) / self.pr(x,0,y)))
        self.x_nb = x.multiply(self._r)

    def fit(self, x, y):
        # Check that X and y have correct shape
        # y = y.values
        # x, y = check_X_y(x, y, accept_sparse=True)
        self._clf = LogisticRegression(solver = self.solver, C=self.C, dual=self.dual, n_jobs=self.n_jobs, random_state=42).fit(self.x_nb, y)
        return self
    
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, penalty='l2', loss='squared_hinge'):
        self.C = C
        self.penalty = penalty
        self.loss = loss

    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def predict(self, x):
        # Verify that model has been fit
        # check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        # check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))
    
    def pre_fit(self, x, y):
        self._r = sparse.csr_matrix(np.log(self.pr(x,1,y) / self.pr(x,0,y)))
        self.x_nb = x.multiply(self._r)

    def fit(self, x, y):
        # Check that X and y have correct shape
        # y = y.values
        # x, y = check_X_y(x, y, accept_sparse=True)
        self._clf = CalibratedClassifierCV(LinearSVC(penalty= self.penalty, C = self.C, loss= self.loss)).fit(self.x_nb, y)

        return self