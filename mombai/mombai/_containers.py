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
    
def as_list(value):
    if value is None:
        return []
    elif is_array(value):
        return list(value)
    else:
        return [value]

def as_array(value):
    if value is None:
        return []
    elif is_array(value):
        return value
    else:
        return [value]

def as_ndarray(value):
    try:
        return np.array(as_list(value))
    except ValueError:
        return np.array(as_list(value), dtype='object')

def as_type(value):
    return value if isinstance(value, type) else type(value)


def as_str(value, max_rows = None, max_chars = None):
    if max_rows is None and max_chars is None:
        return value.__str__()
    else:
        return '\n'.join([row[:max_chars] for row in value.__str__().split('\n')[:max_rows]])

def _eq_attrs(x, y, attrs):
    for attr in attrs:
        if hasattr(x, attr) and not eq(getattr(x, attr), getattr(y, attr)):
            print(attr)
            return False
    return True

def _getitem_as_array(iterable, item, check_bool = True):
    if is_array(item):
        if check_bool and len(item) == len(iterable) and min([isinstance(i, bool) or i in [0,1] for i in item]):
            return [i for i, tf in zip(iterable, item) if tf]
        else:
            return [iterable[i] for i in item]
    else:
        return iterable[item]


def eq(x, y):
    """
    A better nan-handling equality comparison. Here is the problem:
    
    >>> import numpy as np
    >>> assert not np.nan == np.nan  ## What?
    
    The nan issue extends to np.arrays...
    >>> assert list(np.array([np.nan,2]) == np.array([np.nan,2])) == [False, True]
    
    but not to lists...
    >>> assert [np.nan] == [np.nan]
    
    But wait, if the lists are derived from np.arrays, then no equality...
    >>> assert not list(np.array([np.nan])) == list(np.array([np.nan]))
    
    The other issue is inheritance:
    >>> class FunnyDict(dict):
    >>>    def __getitem__(self, key):
    >>>        return 5
    >>> assert dict(a = 1) == FunnyDict(a=1) ## equality seems to ignore any type mismatch
    >>> assert not dict(a = 1)['a'] == FunnyDict(a = 1)['a'] 
    
    >>> import pandas as pd
    >>> import pytest
    >>> from mombai import eq
    
    >>> assert eq(np.nan, np.nan) ## That's better
    >>> assert eq(np.array([np.nan,2]),np.array([np.nan,2]))    
    >>> assert eq(np.array([np.array([1,2]),2]), np.array([np.array([1,2]),2]))
    >>> assert eq(np.array([np.nan,2]),np.array([np.nan,2]))    
    >>> assert eq(dict(a = np.array([np.array([1,2]),2])) ,  dict(a = np.array([np.array([1,2]),2])))
    >>> assert eq(dict(a = np.array([np.array([1,np.nan]),np.nan])) ,  dict(a = np.array([np.array([1,np.nan]),np.nan])))
    >>> assert eq(np.array([np.array([1,2]),dict(a = np.array([np.array([1,2]),2]))]), np.array([np.array([1,2]),dict(a = np.array([np.array([1,2]),2]))]))
    
    >>> assert not eq(dict(a = 1), FunnyDict(a=1))    
    >>> assert eq(1, 1.0)
    >>> assert eq(x = pd.DataFrame([1,2]), y = pd.DataFrame([1,2]))
    >>> assert eq(pd.DataFrame([np.nan,2]), pd.DataFrame([np.nan,2]))
    >>> assert eq(pd.DataFrame([1,np.nan], columns = ['a']), pd.DataFrame([1,np.nan], columns = ['a']))
    >>> assert not eq(pd.DataFrame([1,np.nan], columns = ['a']), pd.DataFrame([1,np.nan], columns = ['b']))
    """
    if x is y:
        return True
    elif isinstance(x, (np.ndarray, tuple, list)):
        return type(x)==type(y) and len(x)==len(y) and _eq_attrs(x,y, ['__shape__']) and np.all(_veq(x,y))
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return type(x)==type(y) and _eq_attrs(x,y, attrs = ['__shape__', 'index', 'columns']) and np.all(_veq(x,y))
    elif isinstance(x, dict):
        if type(x) == type(y) and len(x)==len(y):
            xkey, xval = zip(*sorted(x.items()))
            ykey, yval = zip(*sorted(y.items()))
            return eq(xkey, ykey) and eq(np.array(xval, dtype='object'), np.array(yval, dtype='object'))
        else:
            return False
    elif isinstance(x, float) and np.isnan(x):
        return isinstance(y, float) and np.isnan(y)  
    else:
        try:
            res = x == y
            return np.all(res.__array__()) if hasattr(res, '__array__') else res
        except Exception:
            return False # if you really have no == supported, the two items are not the same

_veq = np.vectorize(eq)

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

def concat(*arrays):
    """joins arrays together efficiently, using np.concatenate if arrays are ndarray, otherwise, using built in sum"""
    arrays = args_to_list(arrays)
    if len(arrays) == 0:
        return []
    elif min([isinstance(arr, np.ndarray) for arr in arrays]):
        return np.concatenate(arrays)
    else:
        return sum([as_list(arr) for arr in arrays], [])