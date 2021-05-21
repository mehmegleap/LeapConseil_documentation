from scipy import stats
from sklearn.feature_selection import *

FEATURES_SELECTOR = dict(
    chi_squared=chi2,
    anova=f_classif,
    pearson=f_regression,
    mutual_info_classif=mutual_info_classif,
    mutual_info_regression=mutual_info_regression,
    low_variance=VarianceThreshold,
    kendaltau=stats.kendalltau,
    spearman=stats.spearmanr
)
