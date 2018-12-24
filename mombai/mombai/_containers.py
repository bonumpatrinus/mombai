from ordered_set import OrderedSet
import numpy as np
import pandas as pd
from mombai._decorators import try_false
from _collections_abc import dict_keys, dict_values
from copy import copy
import sys
version = sys.version_info


if version.major<3:
    def is_array(value):
        return isinstance(value, (list, tuple, range, np.ndarray))
else:
    def is_array(value):
        return isinstance(value, (list, tuple, range, np.ndarray, dict_keys))

def replace(value, what, to = ''):
    what = as_list(what)
    if isinstance(value, str):
        for arg in what:
            while arg in value:
                value = value.replace(arg, to)
    elif is_array(value):
        return type(value)([replace(v, what, to) for v in value])
    elif isinstance(value, dict):
        return type(value)({k: replace(v, what, to) for k, v in value.items()})
    return value    

class ordered_set(OrderedSet):
    """
    An ordered set but one that supports operations with non-sets:
    >>> assert ordered_set([1,2,3]) - 1 == ordered_set([2,3])
    >>> assert ordered_set([1,2,3]) + 4 == ordered_set([1,2,3,4])
    >>> assert ordered_set([1,2,3]) + [3,4] == ordered_set([1,2,3,4])
    >>> assert ordered_set([1,2,3]) & [3,4] == ordered_set([3])
    """
    @classmethod
    def cast(cls, other):
        if isinstance(other, cls):
            return other
        return ordered_set(as_list(other))
    def __str__(self):
        return list(self).__str__()
    def __repr__(self):
        return list(self).__repr__()
    def __add__(self, other):
        return self | ordered_set.cast(other)
    def __and__(self, other):
        return super(ordered_set, self).__and__(ordered_set.cast(other))
    def __sub__(self, other):
        return super(ordered_set,self).__sub__(ordered_set.cast(other))
    def __mod__(self, other):
        return self - (self & other)
    
class slist(list):
    """
    A list of unique items which behaves like an ordered set
    """
    def __init__(self, *args, **kwargs):
        super(slist, self).__init__(ordered_set(*args, **kwargs))
    def __add__(self, other):
        return slist(ordered_set(self) + other)
    def __and__(self, other):
        return slist(ordered_set(self) & other)
    def __or__(self, other):
        return slist(ordered_set(self) | other)
    def __sub__(self, other):
        return slist(ordered_set(self) - other)
    def __mod__(self, other):
        return slist(ordered_set(self) % other)

class nplist(list):
    """
    a list that supports the numpy interface. Does NOT accept bool arrays. This needs to be handled elsewhere
    """
    def __getitem__(self, item):
        me = super(nplist, self).__getitem__
        if (hasattr(item, '__next__ ') or hasattr(item, '__iter__')) and not isinstance(item, str):
            return nplist([me(i) for i in item])
        else:
            return me(item)
    def __mul__(self, other):
        return type(self) * super(nplist,self).__mul__(other)
    def __repr__(self):
        return 'nplist'+ super(nplist,self).__repr__()
    def __str__(self):
        return 'nplist'+ super(nplist,self).__str__()

def as_list(value, tp=list):
    if value is None:
        return []
    elif is_array(value):
        return tp(value)
    else:
        return tp([value])

def as_array(value):
    if value is None:
        return []
    elif is_array(value):
        return value
    else:
        return [value]

def as_ndarray(value):
    if isinstance(value, np.ndarray):
        return value
    value = as_list(value, nplist)
    if len(set([type(v) for v in value]))==1:
        try:
            return np.array(value)
        except ValueError:
            pass
    try:
        return np.array(value, dtype='object')
    except ValueError:
        return value
    raise Exception('could not construct %s as an array' % value)

def as_type(value):
    return value if isinstance(value, type) else type(value)


def as_str(value, max_rows = None, max_chars = None):
    if max_rows is None and max_chars is None:
        return value.__str__()
    else:
        return '\n'.join([row[:max_chars] for row in value.__str__().split('\n')[:max_rows]])




def _args_len(*values):
    lens = slist([len(value) for value in values]) - 1
    if len(lens)>1:
        raise ValueError('all values must have same length')
    return lens[0] if lens else 1
    
def args_len(*values):
    return _args_len(*[as_ndarray(value) for value in values])
    
def args_zip(*values):
    """
    This function is a safer version of zipping. 
    We insist that all elements have size 1 or the same length
    """
    values = [as_list(value) for value in values]
    n = _args_len(*values)
    values = [value*n if len(value)!=n else value for value in values]
    return zip(*values)

def args_to_list(args):
    """
    this is used to allow a function to take a zipped/unzipped parameters:
    >>> from operator import __add__
    >>> from functools import reduce
    >>> def func(*args):
    >>>     args = args_to_list(args)
    >>>     return reduce(__add__, args)
    >>> assert func(1,2,3) == func([1,2,3])
    """
    args = as_list(args)
    return args[0] if len(args)==1 and is_array(args[0]) else args


def args_to_dict(args):
    """
    returns a dict from a list, a single value or a dict.
    >>> assert args_to_dict(('a','b',dict(c='d'))) == dict(a='a',b='b',c='d')
    >>> assert args_to_dict([dict(c='d')]) == dict(c='d')
    >>> assert args_to_dict(dict(c='d')) == dict(c='d')
    >>> assert args_to_dict(['a','b',dict(c='d'), dict(e='f')]) == dict(a='a',b='b',c='d', e='f')
    >>> import pytest
    >>> with pytest.raises(ValueError):
    >>>     args_to_dict(['a','b',lambda c: c]) == dict(a='a',b='b',c='d', e='f')
    """
    args = args_to_list(args)
    res = dict()
    for arg in args:
        if isinstance(arg, dict):
            res.update(arg)
        else:
            if isinstance(arg, str):
                res[arg] = arg
            else:
                raise ValueError('cannot use a non-string %s in the list, it must be assigned a string name by making it into a dict'%arg)
    return res


def _getitem_as_array(iterable, item, check_bool = True):
    if is_array(item):
        if check_bool and len(item) == len(iterable) and min([isinstance(i, bool) or i in [0,1] for i in item]):
            return [i for i, tf in zip(iterable, item) if tf]
        else:
            return [iterable[i] for i in item]
    else:
        return iterable[item]


def concat(*arrays):
    """joins arrays together efficiently, using np.concatenate if arrays are ndarray, otherwise, using built in sum"""
    arrays = args_to_list(arrays)
    if len(arrays) == 0:
        return []
    elif min([isinstance(arr, np.ndarray) for arr in arrays]):
        return np.concatenate(arrays)
    else:
        return sum([as_list(arr) for arr in arrays], [])