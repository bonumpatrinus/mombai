import numpy as np
import pandas as pd
from mombai._decorators import cache
from mombai._containers import as_ndarray

def _eq_attrs(x, y, attrs):
    for attr in attrs:
        if hasattr(x, attr) and not eq(getattr(x, attr), getattr(y, attr)):
            return False
    return True


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

def _cmparr(*arr):
    """
    compare arrays, each of arr is a list of pairs. We stop the moment we have a non zero value
    """
    c = 0
    for a in arr:
        for pair in a:
            c  = cmp(*pair)
            if c!=0:
                break
    return c


def _type(x):
    return str(float if isinstance(x, (int,float, np.int64, np.int32, np.int, np.int16)) else type(x))

def _len(x):
    return len(x) if hasattr(x, '__len__') else 0

def _type_length(x):
    return (_type(x), _len(x))


vtype = np.vectorize(_type)
vlen  = np.vectorize(_len)
vtl = np.vectorize(_type_length)


def cmp(x, y):
    """
    return -1 if x<y else 1 if x>y else 0
    
    This function is a general comparator supporting issues such as nan both on its own and within a list/array
    >>> from numpy import nan
    >>> assert not nan > 1 and not nan < 1 and not nan == 1
    >>> assert not [1,2] > [1,nan] and not [1,2] < [1,nan] and not [1,2] == [1,nan]
    
    >>> assert cmp(nan, 1) == 1 
    >>> assert cmp([1,nan],[1,2]) == 1 

    >>> assert cmp([1,2,3], y = [1,2,3]) == 0
    >>> assert cmp(np.array([1,2,3]), np.array([1,2,3])) == 0
    >>> assert cmp(x = np.array([1,2,nan]), y = np.array([1,2,3])) == 1
    x = np.array([1,2,[1,2,nan]], dtype = 'object')
    y = np.array([1,2,[1,2,3]], dtype = 'object')
    >>> assert cmp(x,y) ==1 
    >>> assert cmp(0,None) == 1 
    >>> assert cmp(0,1) == -1 
    >>> assert cmp(2,1) == 1 
    >>> assert cmp(2,2) == 0 
    >>> assert cmp(2,2.) == 0 
    >>> assert cmp(np.nan,2.) == 1     
    """
    if x is y:
        return 0
    tx = str(float if isinstance(x, (int,float, np.int64, np.int32, np.int, np.int16)) else type(x))
    ty = str(float if isinstance(y, (int,float, np.int64, np.int32, np.int, np.int16)) else type(y))
    if tx<ty:
        return -1
    elif tx>ty:
        return 1
    if isinstance(x, float) and np.isnan(x):
        x = np.inf
    if isinstance(y, float) and np.isnan(y):
        y = np.inf
    if isinstance(x, (np.ndarray, tuple, list)):
        return _cmparr([(len(x), len(y))], zip(x,y))
    elif isinstance(x, dict):
        xkey, xval = zip(*sorted(x.items()))
        ykey, yval = zip(*sorted(y.items()))
        return _cmparr([(len(x), len(y))], zip(xkey, ykey), zip(xval, yval))
    else:
        try:
            return -1 if x<y else 1 if x>y else 0
        except Exception:
            return _cmparr(zip(x,y))

class Cmp(object):
    """
    This function is used to allow comparison between different types, first sorting on type and only then checking for equality within type
    >>> x = [0,'a', None]
    >>> with pytsest.raises(TypeError):
    >>>     sorted(x)
    
    However:
    >>> assert sorted(x, key = Cmp) == [None, 0, 'a']
    """
    def __init__(self, x):
        self.value = x
    
    def __lt__(self, other):
        return self.cmp(other)==-1

    def __gt__(self, other):
        return self.cmp(other)==1

    def __eq__(self, other):
        return self.cmp(other)==0
    
    def cmp(self, other):
        return cmp(self.value, other.value)
    
    def __str__(self):
        return 'Compare ' + self.value.__str__()

    def __repr__(self):
        return 'Compare ' + self.value.__repr__()


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def reverse(values):
    return values[::-1]


def panda_sorter(values):
    cols= list(range(len(values)))
    return pd.DataFrame(data = np.array(values).T, columns = cols).sort_values(by = cols).index.values

vCmp = np.vectorize(Cmp)


def as_cmp(values):
    if isinstance(values, np.ndarray) and len(values.shape)==2:
        values = list(values.T)
    res = []
    for value in values:
        value = as_ndarray(value)
        if len(value.shape)==2:
            res.extend(as_cmp(value))
        elif value.dtype == np.dtype('O'):
            res.append(vCmp(value))
        else:
            res.append(value)
    return res


class Sort(object):
    def __init__(self, values, sorter=np.lexsort, transform=None, key = as_cmp):
        self.values = tuple(as_ndarray(v) for v in values)
        if transform: 
            self.values = tuple(transform(v) for v in self.values)
        self.sorter = sorter
        self.transform = transform
        self.key = key
    
    @property
    @cache
    def keys(self):
        if self.key is None:
            return self.values[::-1]
        else:
            return self.key(self.values)[::-1] 

    @property
    @cache
    def argsort(self):
        return self.sorter(self.keys)
    
    @property
    @cache
    def sorted(self):
        """
        sorting myself based on argsort, returns a transposed matrix!
        """
        return self.sort(self.values)
    
    def sort(self, values):
        """
        sorting values based on argsort
        """
        if isinstance(values, list):
            return [values[i] for i in self.argsort]
        elif isinstance(values, np.ndarray):
            return values[self.argsort]
        elif isinstance(values, tuple):
            return type(values)(self.sort(v) for v in values)
        elif isinstance(values, dict):
            return type(values)({key: self.sort(value) for key, value in values.items()})
    
    def __len__(self):
        return len(self.keys[0])
    
    @property
    def shape(self):
        return (len(self), len(self.keys))

    @property
    @cache
    def _edges(self):
        """
        find the points, in the sorted data, where we have a change in value
        we know the sorted data is indexed by argsort so we then return the coordinates in the original data
        """
        veq = np.vectorize(eq)
        changes = ~np.min(np.array([veq(col[1:], col[:-1]) for col in self.sorted]), axis=0)
        return np.arange(1, len(self))[changes]

    @property
    @cache
    def grouped(self):
        return np.split(self.argsort, self._edges)
        
    def group(self, values):
        """
        based on the keys, group all values
        """
        if isinstance(values, np.ndarray):
            return np.split(self.sort(values), self._edges)
        elif isinstance(values, list):
            return [[values[i] for i in grp] for grp in self.grouped]
        elif isinstance(values, tuple):
            return type(values)(self.group(v) for v in values)
        elif isinstance(values, dict):
            return type(values)({key: self.group(value) for key, value in values.items()})

    @property
    def unique(self):
        mask = np.concatenate([[0], self._edges])
        return [col[mask] for col in self.sorted]
    
    def __str__(self):
        return '%s of %s' % (type(self).__name__, self.values.__str__())
    
    def __repr__(self):
        return '%s of %s' % (type(self).__name__, self.values.__repr__())


#values = [['10', '2', 2, 1, None, np.array([10,2]), np.array([3,4,5]), np.array([3,4])]]
#t = Sort(values, argsort, transform = None, key  = _as_keys)
#t.argsort

        



