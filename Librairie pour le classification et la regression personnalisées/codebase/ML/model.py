from imblearn.ensemble import *
from itertools import combinations
from sklearn.ensemble import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.gaussian_process import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.tree import *
from typing import List, Dict

MODELS = dict(reg=dict(
    ada_boost=AdaBoostRegressor,
    bagging=BaggingRegressor,
    extratrees=ExtraTreesRegressor,
    gradient_boosting=GradientBoostingRegressor,
    random_forest=RandomForestRegressor,
    hist_gradient_boost=HistGradientBoostingRegressor,
    gaussian_process=GaussianProcessRegressor,
    linear_regression=LinearRegression,
    ridge=Ridge,
    kernel_ridge=KernelRidge,
    sdg_regressor=SGDRegressor,
    elastic_net=ElasticNet,
    lars=Lars,
    lasso=Lasso,
    lasso_lars=LassoLars,
    lasso_larsic=LassoLarsIC,
    ardr=ARDRegression,
    bayesian_ridge=BayesianRidge,
    multi_task_elastic_net=MultiTaskElasticNet,
    multi_task_lasso=MultiTaskLasso,
    huber_regressor=HuberRegressor,
    ransac=RANSACRegressor,
    theil_sen=TheilSenRegressor,
    poisson_regressor=PoissonRegressor,
    linear_passive_aggressive=PassiveAggressiveRegressor,
    tweedie_regressor=TweedieRegressor,
    decision_tree=DecisionTreeRegressor,
    extratree=ExtraTreeRegressor,
    linear_svr=LinearSVR,
    nu_svr=NuSVR,
    svr=SVR,
    k_neighbors=KNeighborsRegressor,
    radius_neighbors=RadiusNeighborsRegressor,
    mlp=MLPRegressor),
    classif=dict(
        ada_boost=AdaBoostClassifier,
        bagging=BaggingClassifier,
        extratrees=ExtraTreesClassifier,
        random_forest=RandomForestClassifier,
        gradient_boosting=GradientBoostingClassifier,
        hist_gradient_boost=HistGradientBoostingClassifier,
        linear_passive_aggressive=PassiveAggressiveClassifier,
        logistic_regression=LogisticRegression,
        ridge=RidgeClassifier,
        sgd=SGDClassifier,
        perceptron=Perceptron,
        balanced_bagging=BalancedBaggingClassifier,
        balanced_random_forest=BalancedRandomForestClassifier,
        rusboost=RUSBoostClassifier,
        easy_ens=EasyEnsembleClassifier,
        decision_tree=DecisionTreeClassifier,
        extratree=ExtraTreeClassifier,
        linear_svc=LinearSVC,
        nu_svc=NuSVC,
        svc=SVC,
        k_neighbors=KNeighborsClassifier,
        radius_neighbors=RadiusNeighborsClassifier,
        nearest_centroids=NearestCentroid,
        gaussian_process=GaussianProcessClassifier,
        mlp=MLPClassifier),
    ensemble=dict(reg=dict(
        stacking=StackingRegressor,
        voting=VotingRegressor
    ),
        classif=dict(
            stacking=StackingClassifier,
            voting=VotingClassifier
        )
    )
)


def combine_models(models: List, source: Dict) -> List:
    ensembles, result = [], []

    for length in range(2, len(models) + 1):
        ensembles += list(combinations(models, length))

    for _it in ensembles:
        for ens_name, _methods in source.items():
            _it = list(_it)
            tmp_repr = [_cpl[0] for _cpl in _it]
            ens_repr = ens_name + '+' + '+'.join(tmp_repr)
            result += [((ens_repr, _methods(_it)), 'see combined models args')]
    return result
