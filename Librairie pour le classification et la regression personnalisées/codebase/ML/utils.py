from functools import partial, update_wrapper
from itertools import product
from typing import *


def wrapped_partial(func, **kwargs):
    partial_func = partial(func, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def is_iterable(obj, basestring):
    return bool(obj) and all(isinstance(elem, basestring) for elem in obj)


def is_empty(obj: List) -> List:
    return obj if len(obj) > 0 else [((None, None), None)]


def instantiate_object_from_args(obj_name: str, source: Dict, __call__: Callable = None, **kwargs) -> Any:
    if isinstance(__call__, Callable):
        return __call__(source[obj_name])
    return source[obj_name](**kwargs)


def check_if_none(obj: Union[List, Tuple]) -> List:
    indices = [ind for ind, (i, _) in enumerate(obj) if i is None]
    return [val for ind, val in enumerate(obj) if ind not in indices]


def create_instances(objs: Union[str, Dict, List[str]], source: Dict, **kwargs) -> List[Dict]:
    instances, inputs = [], []
    __call__ = kwargs.get('__call__', None)
    if isinstance(objs, str):
        inputs = list(product(list(source.keys()), ['default']))
    elif is_iterable(objs, dict):
        inputs = [list(obj.items())[-1] for obj in objs]
    elif is_iterable(objs, str):
        inputs = list(product(objs, ['default']))

    for obj in inputs:
        obj_name, args = obj
        parameters = {} if args == 'default' else args
        instances += [((obj_name, instantiate_object_from_args(obj_name=obj_name, source=source, __call__=__call__,
                                                               **parameters)), args)]
    instances = is_empty(instances)
    return instances
