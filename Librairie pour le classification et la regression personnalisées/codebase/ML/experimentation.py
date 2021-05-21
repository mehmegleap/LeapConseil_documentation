import ast
import json
import pandas as pd
from imblearn.pipeline import Pipeline as pipeline_with_sampler
from itertools import chain
from sklearn.model_selection import BaseCrossValidator, ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as pipeline
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

from .config import *
from .cross_validation import pseudo_cross_val_predict
from .scoring import *
from .utils import check_if_none


def run_experiment(cfg: Config, read: Callable,
                   output_dir: str = './experiment/output',
                   **kwargs) -> str:
    res = []
    _check = dict()
    config = cfg
    seed = config.seed
    experiment = config.experiment
    dataset = config.dataset
    task = config.task
    target_col = dataset.target_col
    metrics, _ = list(zip(*config.metrics))
    scorer = dict(metrics)
    samplers, _ = list(zip(*config.samplers))
    data_normalizers, _ = list(zip(*config.scale))
    feature_selectors, _ = list(zip(*config.feature_selection))
    models, models_params = list(zip(*config.models))
    model_names, _ = list(zip(*models))

    cross_validation, cv_params = list(zip(*config.cross_validation))
    cv_names, _ = list(zip(*cross_validation))
    output_file = '%s/%s_%s_res.yaml' % (output_dir, target_col.lower(), experiment)
    use_class_weight = config.use_class_weight
    _check.update({**dict(zip(model_names, models_params)), **dict(zip(cv_names, cv_params))})
    np.random.seed(seed)

    df = read(**asdict(dataset))
    drop_cols = config.drop_col
    if drop_cols is not None:
        columns = set(df.columns).intersection(drop_cols)
        df.drop(labels=list(columns), axis=1)

    encoder = LabelEncoder()
    for var in df.columns:
        if df[var].dtype == "object":
            df[var] = encoder.fit_transform(df[var])

    features, target = df.drop(labels=target_col, axis=1), df[target_col]
    X, y = features.values, target.values
    labels = np.unique(y)

    scorer.update({
        'Acc. label = %s' % str(i): make_scorer(METRICS[task]['class_accuracy'], id_=i) for i in
        range(len(labels)) if task == 'classif'
    })

    configs = list(product(data_normalizers, feature_selectors, samplers, models, cross_validation))

    for conf in configs:
        data_normalizer, feature_selector, sampler, model, cross = conf
        pipe = [data_normalizer, feature_selector, sampler, model]
        param_grid = {
            '%s__k' % feature_selector[0]: list(range(1, X.shape[1] + 1))
        } if feature_selector[0] is not None else {}
        cross_val_name, cv = cross

        pruned_conf = check_if_none(pipe)
        estimator = pipeline(pruned_conf) if not param_grid else pipeline_with_sampler(pruned_conf)

        refit = True if len(metrics) <= 1 else False

        if use_class_weight:
            params = estimator.get_params()
            for p in params:
                if p.endswith('_class_weight'):
                    estimator.set_params(**{p: "balanced"})
                    sampler = ('class_weight_',)


        print('------------%s--------------------------------'%model[0])
        if isinstance(cv, BaseCrossValidator):
            clf = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scorer,
                               cv=cv, verbose=4, refit=refit) #scorer
            clf.fit(X, y)
            scores = clf.cv_results_
            res += [{'data_normalizer': data_normalizer[0], 'sampler': sampler[0], 'model': model[0],
                     'cv': cross_val_name,
                     'cv_args': _check[cross_val_name],
                     'model_args': _check[model[0]],
                     'feat. selection': feature_selector[0], **scores}]


        else:
            param_grid = param_grid if param_grid else {'%s__k' % feature_selector[0]: [None]}
            grid = list(ParameterGrid({
                'estimator': [estimator], **param_grid
            }))

            for param in grid:
                param_res = dict()
                if param['%s__k' % feature_selector[0]] is not None:
                    param_res['param'] = {'%s__k' % feature_selector[0]: param['%s__k' % feature_selector[0]]}
                    estimator.set_params(**param_res['param'])
                try:
                    y_true, y_pred = pseudo_cross_val_predict(estimator, cv, X, y)
                    clf = list(zip(y_true, y_pred))
                    if all(len(yt) == len(yp) == 1 for yt, yp in clf):
                        true_y = list(chain(*y_true))
                        pred_y = list(chain(*y_pred))
                        clf = [(true_y, pred_y)]

                    f = lambda x: [METRICS[task][sc](y_true=x[0], y_pred=x[1]) for sc in scorer]
                    scores = list(map(f, clf))
                    scores = [list(scorer.keys())] + scores
                    tmp_scores = list(zip(*scores))
                    scores = {'split%s_test_%s' % (i, met[0]): met[i] for met in tmp_scores for i in range(len(met[1:]))}
                    scores.update({
                        'mean_test_%s' % met[0]: np.array(list(met[1:])).mean() for met in tmp_scores})
                    scores.update({
                        'std_test_%s' % met[0]: np.array(list(met[1:])).std() for met in tmp_scores})

                    res += [{'data_normalizer': data_normalizer[0], 'sampler': sampler[0], 'model': model[0],
                             'cv': cross_val_name,
                             'cv_args': _check[cross_val_name],
                             'model_args': _check[model[0]],
                             'feat. selection': feature_selector[0], **scores}]
                except Exception as e:
                    print('Failure - %s'%(str(e)))

    dump(res, output_file)
    return output_file


def display_result(filepath: str, dispatch : bool = True) -> pd.DataFrame:
    res = load(filepath)
    scores = res
    if dispatch:
        scores = []
        for each in res:
            metrics_ = list(each.keys())
            test_metrics_ = []
            test_scores_ = []
            for metric in metrics_:
                if ('_test_' in metric) or ('param' in metric) or ('mean_' in metric) or ('std_' in metric):
                    test_metrics_ += [metric]
                    add = each[metric].data.tolist() if isinstance(each[metric], np.ndarray) else each[metric]
                    test_scores_ += [add]
                    del each[metric]
            dispatched = list(zip(*test_scores_))
            f = [list(zip(test_metrics_, cls)) for cls in dispatched]
            f = [{**each, **dict(_f)} for _f in f]
            scores += f

    df = pd.DataFrame(scores)
    return df
