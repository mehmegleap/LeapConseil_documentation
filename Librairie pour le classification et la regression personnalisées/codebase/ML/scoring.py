import numpy as np
from sklearn.metrics import *
from typing import List, Dict

from .utils import wrapped_partial


def id_accuracy_score(y_true, y_pred, id_: int) -> float:
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    r = cm.diagonal()
    return r[id_]


METRICS = dict(
    reg=dict(
        EVS=explained_variance_score,
        MAXErr=max_error,
        MAE=mean_absolute_error,
        MSE=mean_squared_error,
        LMSE=mean_squared_log_error,
        DAR=median_absolute_error,
        MAPE=mean_absolute_percentage_error,
        R2=r2_score,
        MPD=mean_poisson_deviance,
        MGD=mean_gamma_deviance,
        MTD=mean_tweedie_deviance,
    ),
    classif=dict(
        Accuracy=accuracy_score,
        AUC=auc,
        AP=average_precision_score,
        balancedAcc=balanced_accuracy_score,
        F1=f1_score,
        precision=precision_score,
        Recall=recall_score,
        class_accuracy=id_accuracy_score
    )
)

for name, metric in [('precision', precision_score),
                     ('Recall', recall_score), ('F1', f1_score),
                     ('jaccard', jaccard_score)]:
    SCORERS = METRICS['classif']
    SCORERS[name] = wrapped_partial(metric, average='binary')
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = wrapped_partial(metric, pos_label=None,
                                                  average=average)
