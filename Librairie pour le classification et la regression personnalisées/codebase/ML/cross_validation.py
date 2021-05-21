import numpy as np
import pandas as pd
from sklearn.model_selection import *
from typing import Tuple


class TrainTestCV:
    def __init__(self, train_percent, test_size=0.2, random_state=42):
        super(TrainTestCV, self).__init__()
        self.test_size = test_size
        self.train_percent = train_percent
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: np.ndarray = None, groups: np.ndarray = None):
        indices = np.arange(len(y))
        train, test = train_test_split(indices, test_size=self.test_size, random_state=self.random_state, stratify=y)
        train = np.random.choice(train, int(len(train) * self.train_percent), replace=False)
        yield train, test


class PseudoLeaveOneOut:
    def __init__(self, p_choices: int = 10):
        super(PseudoLeaveOneOut, self).__init__()
        self.p_choices = p_choices
        self.n_splits = 0

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(len(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        labels = np.unique(y)
        selected_test_indices = []
        n_choices = self.p_choices
        for label in labels:
            index = np.argwhere(y == label)
            index = index.flatten()
            test_indices = np.random.choice(index, min(n_choices, len(index) // 2), replace=False)
            selected_test_indices += test_indices.tolist()

        self.n_splits = len(selected_test_indices)
        return selected_test_indices

    def get_n_splits(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
        return self.n_splits


CROSS_VALIDATION = dict(
    group_kfold=GroupKFold,
    group_shuffle_split=GroupShuffleSplit,
    kfold=KFold,
    leave_one_group_out=LeaveOneGroupOut,
    leave_pgroups_out=LeavePGroupsOut,
    leave_one_out=LeaveOneOut,
    predefined_split=PredefinedSplit,
    repeat_kfold=RepeatedKFold,
    repeat_stratified_kfold=RepeatedStratifiedKFold,
    shuffle_split=ShuffleSplit,
    stratified_kfold=StratifiedKFold,
    stratified_shuffle_split=StratifiedShuffleSplit,
    timeseries_split=TimeSeriesSplit,
    pseudo_leave_one_out=PseudoLeaveOneOut,
    train_test=TrainTestCV,
)


def pseudo_cross_val_predict(model, cv, X: np.ndarray, y: np.ndarray) -> Tuple:
    y_pred, y_true = [], []
    splits = cv.split(X, y, groups=None)
    n_splits = cv.n_splits if hasattr(cv, 'n_splits') else 1
    print('Fitting %s folds for each of 1 candidates, totalling %s fits'%(n_splits, n_splits))
    ind = 1
    for train_ix, test_ix in splits:
        print('[CV %s/%s] .............'%(ind, n_splits), end='')
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        y_true.append(y_test.tolist())
        y_pred.append(yhat.tolist())
        print('END ')
        ind += 1
    return y_true, y_pred
