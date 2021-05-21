from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import *
from typing import Any

SCALERS = dict(
    max_abs=MaxAbsScaler,
    min_max=MinMaxScaler,
    robust=RobustScaler,
    normalizer=Normalizer,
    poly=PolynomialFeatures,
    power=PowerTransformer,
    quantile=QuantileTransformer,
    standard=StandardScaler,
    variance_threshold=VarianceThreshold # a enlever
)

