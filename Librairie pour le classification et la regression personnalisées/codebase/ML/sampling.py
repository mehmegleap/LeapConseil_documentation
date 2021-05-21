from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import *

SAMPLERS = dict(
    over=RandomOverSampler,
    smote=SMOTE,
    adasyn=ADASYN,
    boderline_smote=BorderlineSMOTE,
    smotenc=SMOTENC,
    under=RandomUnderSampler,
    centroid=ClusterCentroids,
    condensed_nearest_neighbour=CondensedNearestNeighbour,
    edited_nearest_neighbour=EditedNearestNeighbours,
    repeated_edited_nearest_neighbour=RepeatedEditedNearestNeighbours,
    all_knn=AllKNN,
    instance_hardness_threshold=InstanceHardnessThreshold,
    neighbour_hood_cleaning_rule=NeighbourhoodCleaningRule,
    one_side_selection=OneSidedSelection,
    tomek_link=TomekLinks,
    nearmiss=NearMiss,
    smotenn=SMOTEENN,
    smotetomek=SMOTETomek)


def instantiate_imblearn_sampler_from_args(sampler: str, **kwargs):
    return SAMPLERS[sampler](**kwargs)
