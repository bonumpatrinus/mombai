from _collections_abc import dict_keys
from mombai._decorators import decorate, try_back, support_kwargs, relabel, cache
from mombai._compare import Cmp, eq, Sort
from mombai._containers import as_ndarray, as_list, args_zip, _args_len, args_to_list, args_to_dict, slist , _getitem_as_array, concat, as_str, nplist
from mombai._dict_utils import dict_apply, dict_zip, dict_concat, dict_merge, data_and_columns_to_dict, items_to_tree, _pattern_to_item, _is_pattern, _as_pattern
from mombai._dict import Dict
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from functools import partial
#from tqdm import tqdm

def _value(_):
    return _.value

def cartesian(*arrays):
    """
    Creates a cartesian product of multiple arrays, we only use it for 2-d cartesian product in practice
    """
    arrays = args_to_list(arrays)
    shape = (len(x) for x in arrays)
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    return ix

def hstack(value):
    return np.asarray(value).T if len(value)>1 else value[0]

vstack = concat

_str_5x50 = partial(as_str, max_rows = 5, max_chars = 50)


def _max_width(txt, max_width = None):
    if not max_width:
        return txt
    rows = txt.split('\n')
    rng = range(0, max(map(len, rows)), max_width)
    return '\n\n'.join(['\n'.join([row[n : n+max_width] for row in rows]) for n in rng])

class Dictable(Dict):
    """
    Dict is a "calculation graph" allowing us to interactively calculate additional values and store output in the Dict
    Dictable is both a dict and a table: it is a Dict of numpy.ndarray that allows quick parallelisation of Dict call operations.
    In addition, we support "table" operations such as join/sort/filter table and 2-d access
    t = Dictable(a = [1,2,3])
    
    """
    def __init__(self, data=None, columns=None, **kwargs):
        """
        >>> d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
        >>> assert len(d) == 3
        >>> assert isinstance(d, dict)
        >>> assert np.allclose(d.a, [1,2,3])
        >>> assert np.allclose(d.b, [2,2,2])
        >>> assert isinstance(d.b, np.ndarray)
        """
        kwargs.update(data_and_columns_to_dict(data,columns))
        super(Dictable, self).__init__(kwargs)
        for key, value in self.items():
            super(Dictable, self).__setitem__(key, as_ndarray(value))
        n = len(self)
        for key, value in self.items():
            self.__setitem__(key, value, n)

    def __len__(self):
        """
        >>> import pytest
        >>> d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
        >>> assert len(d) == 3
        >>> with pytest.raises(ValueError):
        >>>     d = Dictable(a = [1,2,3], b=2, c=[3,4])
        """
        return _args_len(*self.values())

    @property
    def shape(self):
        """
        >>> import pytest
        >>> d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
        >>> assert d.shape == (3,3)
        >>> d.d = 1
        >>> assert d.shape == (3,4)
        
        """
        return (len(self), super(Dictable, self).__len__())

    def __iter__(self):
        """
        >>> d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
        >>> iters = [row for row in d]
        >>> assert iters == [{'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 2, 'c': 4}, {'a': 3, 'b': 2, 'c': 6}]
        """
        for i in range(len(self)):
            yield self[i]
            
    def _bool2mask(self, mask, exc = False, check_bool=True):
        if check_bool and (hasattr(mask, '__next__ ') or hasattr(mask, '__iter__')) and not isinstance(mask, str) and hasattr(mask, '__len__') and len(mask) == len(self) and min([isinstance(i, bool) or i in [0,1] for i in mask]):
            mask = np.array(mask)
            if exc:
                mask = ~mask
            return np.arange(len(self))[mask]
        else:
            return as_ndarray(mask)

    def _mask(self, mask, exc=False, check_bool = True):
        """ 
        Applying a mask to each of the nd.array values in the dict
        >>> d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
        >>> subset = d._mask([True, True, False])
        >>> assert len(subset) == 2
        >>> complement = d._mask([True, True, False], True)
        >>> assert len(complement) == 1        
        >>> resampled = d._mask([2,1,0,0,2])
        >>> assert len(resampled)==5 and np.allclose(resampled.a, [3,2,1,1,3]) 
        
        What is check_bool?
        >>> self = Dictable(a = 1)
        How do we understand self._mask([0]) ?
            (a) is [0] the INDEX and we should return self
            (b) it is a boolean and we should return nothing?
            
        what about Dictable(a = [1,2])._mask([0,1])?
            (a) is it the index and we return self
            (b) it is a bool and we return the last row?
        
        if check_bool = True (default) then we assume mask is boolean first. If check_bool = False, then we actually don't check for bool at all.
        
        """
        if isinstance(mask, slice):
            if exc:
                index = np.array([True] * len(self)) 
                index[mask] = False
                mask = index
        else:
            mask = self._bool2mask(mask, check_bool, exc)
            if exc and len(mask) == 0: ## include everything, as mask be ufunc invert
                return self 
        return type(self)({key : value[mask] for key, value in self.items()})

    def _subset(self, mask):
        """
        _mask and _subset are almost identical. The main problem is when self has 1 or 2 elemets.
        self = Dictable(a = 1)
        How do we understand self._mask([0]) ?
            (a) is [0] the INDEX and we should return self
            (b) it is a boolean and we should return nothing?
            
        what about Dictable(a = [1,2])._mask([0,1])?
            (a) is it the index and we return self
            (b) it is a bool and we return the last row?        
        """
        
    def get(self, key, *value):
        """
        The generic return is for a column of None's  of the same length
        >>> d = Dictable(a = [1,2,3])
        >>> assert list(d.get('b')) == [None, None, None]
        We do not use value = None signature as the default None depends on the size of self
        """
        if key in self:
            return self[key]
        else:
            if len(value):
                return value[0]
            else:
                return [None]*len(self)


    def __setitem__(self, key, value, n = None):
        """
        We check for length and ndarray nature of each value and the assign it to the dict.
        >>> import pytest
        >>> d = Dictable(a = [1,2,3,4,5])
        >>> assert isinstance(d.a, np.ndarray)
        >>> d['b'] = 1
        >>> assert isinstance(d.b, np.ndarray) and len(d.b) == 5
        >>> with pytest.raises(ValueError):
        >>>     d['c'] = [1,2]
        >>> d = Dictable()
        >>> d['a'] = 1 ## this should work even though len(self)==0
        >>> with pytest.raises(ValueError):
        >>>     d['b'] = [1,2]
        
        >>> d = Dictable(fn = ['//root/fn1.csv', '//root/fn2.csv'])
        >>> d['path','file'] = d[lambda fn: [_ for _ in fn.split('/') if _]]

        """
        value = as_ndarray(value)
        n = n or len(self)
        if len(value)==n or len(self.keys()) == 0:
            pass
        elif len(value)== 1:
            value = value * n if isinstance(value, list) else np.concatenate([value] * n)
        else:
            raise ValueError('cannot set item of mismatched length %s to array of size %s'%(len(value), n))
        if isinstance(key, tuple):
            for k, v in zip(key, zip(*value)):
                super(Dictable, self).__setitem__(k, v)
        else:
            super(Dictable, self).__setitem__(key, value)

    def _vectorize(self, function, relabels=None):
        """
        This function try to line-by-line running 
        >>> d = Dictable(a = [1,2,3,4,5])
        >>> d.b = d[lambda a: range(a)]
        >>> vsum = d._vectorize(sum)
        >>> import pytest
        >>> import numpy as np
        >>> with pytest.raises(TypeError):
        >>>     sum(d.b)
        >>> assert np.allclose(vsum(d.b), [0,1,3,6,10]) ## triangular functions        
        """
        def wrapped(*args, **parameters):
            args_ = list(args_zip(*args))
            kwargs_ = dict_zip(parameters)
            if len(args_)>0 and len(kwargs_)>0:
                res = [function(*a, **k) for a, k in zip(args_, kwargs_)]
            elif len(args_)>0:
                res = [function(*a) for a in  args_]
            elif len(kwargs_)>0:
                res = [function(**k) for k in kwargs_]
            else:
                res = []
            return res
        return try_back(decorate)(wrapped, function)

    _precall = _vectorize
    
    def _inc_or_exc(self, exc, *functions, **filters):
        res = self.copy()
        for function in args_to_list(functions):
            res = res._mask(res.apply(function), exc)
        for key, value in filters.items():
            value = as_list(value)
            @self._precall
            def function(x):
                  return x in value
            res = res._mask(function(res[key]), exc)
        return res
    
    def exc(self, *functions, **filters):
        """
        excludes values from table. can be done either as direct filter:
        >>> d = Dictable(a = range(10))
        >>> assert list(d.exc(a = [1,2,3]).a) == [0,4,5,6,7,8,9]
        
        Or can exclude based on functions of parameters:
        >>> d = Dictable(a = range(10),  b = [2,1]*5)
        >>> res = d.exc(lambda a,b: b<a, a = range(5,10))
        >>> assert list(res.a) == [0,1,2] and list(res.b) == [2,1,2]
        """
        return self._inc_or_exc(True, *functions, **filters)
    
    def inc(self, *functions, **filters):
        """
        Includes values from table. can be done either as direct filter:
        >>> d = Dictable(a = range(10))
        >>> assert len(d.inc(a = [1,2,3])) == 3
        >>> assert list(d.inc(a = [1,2,3]).a) == [1,2,3]
        
        Or can include based on functions as criteria:
        >>> d = Dictable(a = range(10),  b = [2,1]*5)
        >>> res = d.inc(lambda a,b: b<a, a = range(5))
        >>> assert list(res.a) == [3,4] and list(res.b) == [1,2]
        """
        return self._inc_or_exc(False, *functions, **filters)
    
    def __getitem__(self, item):
        """
        >>> from mombai import *
        >>> d = Dictable(a = range(10), b=[1,2]*5)
        
        accessing using a mask or a range:
        >>> assert list(d[d.a>5].a) == [6,7,8,9]
        >>> assert list(d[np.array([0,1])].a) == [0,1]
        >>> assert list(d[range(2)].a) == [0,1]

        accessing using an int:
        >>> assert d[0] == dict(a=0, b=1)
        >>> assert d[-1] == dict(a=9, b=2)
        
        filtering using a dict
        >>> res = d[dict(a = [1,2])]
        >>> assert list(res.a) == [1,2]

        2-d access
        >>> res = d[d.a>5, 'a'] 
        >>> assert list(res) == [6,7,8,9]
        >>> res = d[d.a>5, ['a','b']]
        >>> assert isinstance(res, Dictable) and res.shape == (4,2) and list(res.a) == [6,7,8,9]
        >>> res = d[d.a>5, lambda a, b: a+b]
        >>> assert list(res) == [6+1,7+2,8+1,9+2]
        
        access using normal Dict access
        >>> res = d['a']
        >>> assert list(res) == list(range(10))
        >>> res = d['a','b']
        >>> assert isinstance(res, list) and len(res)==2 and list(res[0]) == list(range(10))
        >>> res = d[lambda a,b: a % b]
        >>> assert list(res) == [0,1]*5
        
        access as tree:
        >>> d = Dictable(name = ['abe', 'beth'], gender = ['m','f'])
        >>> assert d['%name/%gender'] == dict(abe = 'm', beth = 'f')
        """
        if isinstance(item, (np.ndarray,slice, range)):
            return self._mask(item)
        elif isinstance(item, int):
            return Dict({key : value[item] for key, value in self.items()})
        elif isinstance(item, dict):
            return self.inc(**item)
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (int, slice, dict, np.ndarray, range)):
            return self[item[0]][item[1]]
        elif isinstance(item, (list, dict_keys)):
            return type(self)(super(Dictable, self).__getitem__(item))
        elif _is_pattern(item):
            return self.to_tree(item)
        else:
            return super(Dictable, self).__getitem__(item)
    
    @classmethod
    def concat(cls, *others):
        """
        Unlike pd.concat, is a simple extension of dict_concat.
        1) the resulting keys are the union of all keys 
        2) For each key, the output is a list of np.arrays from each table, and of length of original table
        3) We then apply a vertical stacking using np.concatenate
    
        If you have two Dictables that you want to merge horizontally, use e.g. lhs.merge(rhs), lhs.left_join(rhs) etc

        >>> from mombai import *
        >>> others = (Dictable(a=3, b= ['a','b'], c=1),Dictable(a=[1,2,3,], b=[[5]], d='hi'))
        >>> res = Dictable.concat(*others)
        >>> assert list(res.d) == [None] * 2 + ['hi'] * 3  
        >>> assert list(res.c) == [1] * 2 + [None] * 3        
        """
        others = args_to_list(others)
        others = [cls(other) for other in others]
        concated = dict_concat(others)
        merged = dict_apply(concated, concat)
        return cls(merged)

    def __add__(self, other):
        """
        Adding together two tables will be done vertically, with the keys being the union of both tables and None entered where one table has a key and the other does not:
        
        >>> from mombai import *    
        >>> x = Dictable(a = [1,2,3], b=[4,5,6])
        >>> y = Dictable(a = [1,2,3], c=[4,5,6])
        >>> z = x+y
        >>> assert sorted(z.keys()) == ['a','b','c']
        >>> assert list(z.a) == [1,2,3,1,2,3] and list(z.c) == [None, None, None, 4, 5, 6] and list(z.b) == [4, 5, 6, None, None, None]
        """
        return Dictable.concat(self, other)
        
    def __sub__(self, other):
        return Dictable({key : value for key, value in self.items() if key not in as_list(other)})

    def PrettyTable(self, *args, **kwargs):
        if len(self) == 0:
            return self.keys() ## pretty table does not print column names if there are no rows.
        x = PrettyTable(*args, **kwargs)
        x.field_names = self.keys()
        for row in self:
            x.add_row(row.values())
        return x
    
    def __str__(self, max_rows=None, max_width = 200):
        if not max_rows or len(self)<=abs(max_rows):
            return _max_width(self.do(_str_5x50, self.keys()).PrettyTable(border=False).__str__(), max_width)
        else:
            top = self[:max_rows] if max_rows>0 else self[max_rows:]
            pt = _max_width(top.do(_str_5x50, self.keys()).PrettyTable(border=False).__str__(), max_width)
            return '\n'.join([pt ,'...%i rows...'%len(self)])

    def __repr__(self):
        return 'Dictable[%s x %s] '%self.shape + '\n%s'%self.__str__(5)
    
    def _Sort(self, *keys):
        return Sort([self[key] for key in args_to_list(keys)])

    def sort(self, *by):
        """
        >>> from mombai import *
        >>> import numpy as np
        >>> d = Dictable(a = [_ for _ in 'abracadabra'], b=range(11), c = range(0,33,3))
        >>> d.c = np.array(d.c) % 11
        >>> res = d.sort('c')
        >>> assert list(res.c) == list(range(11))
        
        >>> res = d.sort('a','c')
        >>> print(res)
        >>> assert ''.join(res.a) == 'aaaaabbcdrr' and list(res.c) == [0,4,8,9,10] + [2,3] + [1] + [7] + [5,6]
        
        >>> d = d.sort(lambda b: b*3 % 11) ## sorting again by c but using a function
        >>> assert list(d.c) == list(range(11))
        """
        return self._Sort(*by).sort(self)

    def listby(self, *by, **kwargs):
        """
        >>> d = Dictable(a = [_ for _ in 'abracadabra'], b=range(11), c = [_ for _ in 'harrypotter'])
        
        List by a single key:
        >>> per_a = d.listby('a')
        >>> assert list(per_a.a) == ['a','b','c','d','r'] and list(per_a.b[0]) == [0,3,5,7,10]
        
        or list by multiple keys
        >>> per_ac = d.listby('a', 'c')
        >>> assert per_ac.keys() == ['a','c','b']
        >>> assert len(per_ac) == 10 and list(per_ac.inc(a = 'a', c='r').b[0]) == [3,10] and list(per_ac.c[:4]) == ['h','p','r','t']
        
        We can list by values which we create and assign to new keys on the fly:
        >>> res = d.listby(dict(bmod2 = lambda b: b % 2), 'a')
        >>> assert res.keys() == ['bmod2','a','b','c']
        >>> assert res[0].do(list, 'b','c') == Dict(bmod2=0, a = 'a', b = [0,10], c = ['h','r']) ## top row
        """
        keys2values = args_to_dict(by)
        keys2values.update(kwargs)
        keys = slist(keys2values.keys())
        values = list(keys2values.values())
        non_keys = self.keys() - keys
        idx = self._Sort(*values)
        res = type(self)(zip(keys, idx.unique))
        for key in non_keys:
            res[key] = idx.group(self[key])
        return res
    
    def groupby(self, *by, grp = 'grp', **kwargs):
        """
        We group together all records that share the same keys determined by "by". 
        The records form a mini Dictable which is available in the grp key.
        
        Example:  
        >>> from mombai import *
        >>> d = Dictable(a = [_ for _ in 'abracadabra'], b=range(11), c = [_ for _ in 'harrypotter'])
        
        Group by a single key:
        >>> per_a = d.groupby('a')
        >>> assert list(per_a.a) == ['a','b','c','d','r'] and list(per_a.grp[0].b) == [0,3,5,7,10]
        
        or group by multiple keys
        >>> per_ac = d.groupby('a', 'c')
        >>> assert per_ac.keys() == ['a','c', 'grp']
        >>> assert len(per_ac) == 10 and list(per_ac.inc(a = 'a', c='r').grp[0].b) == [3,10] and list(per_ac.c[:4]) == ['h','p','r','t']
        
        We can group by values which we create and assign to new keys on the fly:
        >>> res = d.groupby(dict(bmod2 = lambda b: b % 2), 'a', grp = 'table')
        >>> assert res.keys() == ['bmod2','a','table']
        >>> assert list(res[0].table.b) == [0,10] ## top row
        self=d; by = 'a'; kwargs = {}
        """
        keys2values = args_to_dict(by)
        keys2values.update(kwargs)
        keys = slist(keys2values.keys())
        values = list(keys2values.values())
        non_keys = self.keys() - keys
        idx = self._Sort(*values)
        res = type(self)(zip(keys, idx.unique))
        res[grp] = as_ndarray([type(self)(zip(non_keys, row)) for row in zip(*[idx.group(self[key]) for key in non_keys])])
        return res
    
    def unlist(self):
        tables = [type(self)(row) for row in self]
        return type(self)({key : np.concatenate([table[key] for table in tables]) for key in self.keys()})
    
    def ungroup(self, grp = 'grp'):
        if len(self) == 0:
            return self
        tables = list([type(self)(Dict(row[grp]) + (row-grp)) for row in self])
        keys = tables[0].keys()
        return type(self)({key : np.concatenate([table[key] for table in tables]) for key in keys})
        
    def update(self, other):
        """
        We need to override the update method since update does not size-verify the updates
        >>> import pytest
        >>> x = Dictable(a = [1,2,3])
        >>> y = Dictable(b = [1,2])
        >>> with pytest.raises(ValueError):
        >>>     x.update(y)
        """
        for k, v in other.items():
            self[k] = v
        
    def pivot_table(self, index, columns, values, aggfunc = None):
        """
        Creates a pivot table with x and y being existing columns (x can be multiple columns) and z is a column/calculated function
        >>> from mombai import *
        >>> self = Dictable(x = [1,2,3,1,2,3,1,2,3], y = [4,4,4,5,5,5,6,6,6])
        >>> pt = self.pivot_table(index='x',columns='y',values=lambda x,y: x*y, aggfunc=sum)
        >>> assert pt == Dictable({'x': [1, 2, 3], '4': [ 4,  8, 12], '5': [ 5, 10, 15], '6': [ 6, 12, 18]})
        """
        x = args_to_dict(index)
        y = args_to_dict(columns)
        assert len(y) == 1, 'Cannot have multiple values columns'
        keys = list(x.keys()) + list(y.keys())
        vals = list(x.values()) + list(y.values())        
        gb = self._Sort(*vals)
        res = type(self)(zip(keys, gb.unique))
        z = Dictable(z = gb.group(self[values])).do(aggfunc, 'z')
        gbx = res._Sort(*x.keys())
        rtn = type(self)(zip(x.keys(), gbx.unique))
        yvals = gbx.group(res[columns])
        zvals = gbx.group(z.z)
        dicts = {str(key): value for key, value in dict_concat([dict(zip(yval, zval)) for yval, zval in zip(yvals, zvals)]).items()}
        rtn.update(dicts)
        return rtn
    
    def unpivot(self, index, columns, values):
        """
        index is a list/single key, on which the data is pivoting
        columns is a string, and is the name of the key describing the current column names
        values is also a string, and is the name of the data underneath the columns
        >>> d = Dictable(a = range(5))        
        >>> for b in range(5, 10):
        >>>     d[str(b)] = d.a * b
        >>> res = d.unpivot('a', 'b', 'axb').do(lambda value: int(value), 'b')
        >>> assert list(res.a * res.b) == list(res.axb)
        """
        
        index = as_list(index)
        yvals = self.keys() - index  
        n = len(yvals)
        m = len(self)
        res = type(self)({key : concat([self[key]]*n) for key in index})
        res[columns] = sum([[y] * m for y in yvals], [])
        res[values] = concat([self[y] for y in yvals])
        return res
 
    def _on_left_and_on_right(self, other, on_left=None, on_right=None):
        """
        quick helper function for keys
        """
        on_left = self.keys() & other.keys() if on_left is None else as_list(on_left)
        on_right = on_left if on_right is None else as_list(on_right)            
        return on_left, on_right
    
    def pair(self, other, on_left=None, on_right=None):
        """
        This function the heavy lifting of creating pairing of the left/right indices that match on keys. 
        returns a Dictable with lhs_idx and rhs_idx containing the paired indices
        >>> self  = lhs = Dictable(a = [1,2,3,4], b=[1,2,1,2], c=[1,1,2,2])
        >>> other = rhs = Dictable(a = [4,3,2,1], b=[1,2,1,2], c=[1,1,2,2])
        >>> res = lhs.pair(rhs, ['b','c']) ## join on identical columns!
        >>> assert list(map(list, res.idx)) == [[0, 0 + 4], [1, 1 + 4], [2, 2 + 4], [3, 3+4]] 
        >>> res = lhs.pair(rhs, 'a') ## join reversed columns
        >>> assert list(map(list, res.lhs_idx)) == [[0], [1], [2], [3]] and  list(map(list, res.rhs_idx)) == [[3], [2], [1], [0]] 
        >>> res = lhs.pair(rhs, []) ## full cross join. 
        >>> assert len(res) == 1 and res.lhs_len[0] == len(lhs) and res.rhs_len[0] == len(rhs)        

        >>> from mombai import Dictable
        >>> self = Dictable(a=1)
        >>> other = Dictable(b=2)
        >>> pair = self.pair(other)
        >>> assert len(pair) == 1 and len(pair.idx[0]) == len(self) + len(other)

        self = Dictable(a = [1,1,2,2,3,3])
        other = Dictable(a = [1,1,1,2,3,4])
        on_left = on_right = None
        self = Dictable(a=1); other=Dictable(a=[1,2])
        self.pair(other)
        """
        on_left, on_right = self._on_left_and_on_right(other, on_left, on_right)
        lhs_len = len(self)
        joint_keys = [np.concatenate((self[left_key], other[right_key])) for left_key, right_key in list(zip(on_left, on_right))]
        idx = Sort(joint_keys)
        res = Dictable(idx = idx.grouped)
        res = res(lhs_idx = lambda idx: idx[idx<lhs_len])(rhs_idx = lambda idx: idx[idx>=lhs_len]-lhs_len)
        res = res(lhs_len = lambda lhs_idx: len(lhs_idx))(rhs_len = lambda rhs_idx: len(rhs_idx))
        return res
    
    def _join(self, pair, other, on_left, on_right, merge='a'):
        """
        This function takes a pairing and then performs several actions:
        1) resample each pair into a cartesian product
        2) resample using _mask each of self and other into two equal length tables
        3) merge the two tables using dict_merge. Here we pair together any columns that repeat on both self and other, with the exception of columns involved in the join the on_left/on_right that we know are identical
        >>> from mombai import *
        >>> lhs = Dictable(a = [1,2,3,4], b=[1,2,1,2], c=[1,1,2,2], d='d')
        >>> rhs = Dictable(a = [4,3,2,1], b=[1,2,1,2], c=[1,1,2,2], e='e')
        >>> pair = lhs.pair(rhs, ['b','c']) ## join on identical columns!
        >>> res = lhs._join(pair, rhs, on_left=['b','c'], on_right=['b','c'])
        >>> assert list(map(list, res.a)) == [[1, 4], [2, 3], [3, 2], [4, 1]]
        >>> assert set(res.d) == {'d'} and set(res.e) == {'e'}       
        """
        res = pair.exc(rhs_len=0).exc(lhs_len=0)
        res = res(pairs = lambda lhs_idx, rhs_idx: cartesian(lhs_idx, rhs_idx))
        lhs_idx, rhs_idx = np.concatenate(res.pairs).T
        dicts = [self._mask(lhs_idx, check_bool = False), other._mask(rhs_idx, check_bool = False)]
        duplicate_columns = [left for left, right in zip(on_left, on_right) if left==right and left in self]
        merged = dict_merge(dicts, policy = merge, dict_type = dict, policies = {col : 'left' for col in duplicate_columns})
        return type(self)(dict_apply(merged, hstack, {col : None for col in duplicate_columns}))

    def _left_xor(self, pair, other):
        res = pair.inc(rhs_len=0).exc(lhs_len=0)
        lhs_idx = sorted(concat(res.lhs_idx))
        return self._mask(lhs_idx, check_bool = False)

    def _right_xor(self, pair, other):
        res = pair.exc(rhs_len=0).inc(lhs_len=0)
        rhs_idx = sorted(concat(res.rhs_idx))
        return other._mask(rhs_idx, check_bool = False)

    def merge(self, other, on_left=None, on_right=None, merge='a'):
        """
        Dictable.merge is similar to pd.merge we perform an inner join based on on_left and on_right
        Unlike pandas.merge, on_left and on_right need not be actual columns:
            
        >>> self = Dictable(name = ['James', 'Adam', 'Rosalyn'], surname = ['Maxwell', 'Smith', 'Franklin'], grades = [95, 89, 100], subject = ['Physics', 'Economics', 'Chemistry'])
        >>> other = Dictable(name = ['JAMES', 'GEORGE', 'ROS'], surname = ['MAXWELL', 'WASHINGTOM', 'FRANKLIN'], nationality = ['UK', 'US', 'UK'], grades = [92, 96, 76])
        >>> pair = self.pair(other, on_left = [lambda name: name[:3].lower(), lambda surname: surname.lower()], on_right=None)        
        
        We see the pair function has matched Maxwell and Franklin even though the original data has difference lower/upper cases.
        
        Once the pairing has been done, we end up with two tables of equal length. Now we need to merge the two Dictables using dict_merge policy.
        The default behaviour is appending: if the same column name appears in both:
        >>> merged = self.merge(other, [lambda name: name[:3].lower(), lambda surname: surname.lower()])        
        >>> assert list(map(list, merged.name)) == [['Rosalyn', 'ROS'], ['James', 'JAMES']] 
        >>> assert list(map(list, merged.grades)) == [[100, 76], [95, 92]] 
        We can then decide what to do:
        >>> merged.do(np.mean, 'grades').do(lambda value: value[0], 'name', 'surname')
        >>> merged(avg = np.mean(merged.grades, axis=1))
        
        To summarise, we split the inner_join function into three parts:
            1) pairing of the two tables (using self.pair)
            2) subsets selection/cartesian product from the two tables
            3) dict_merge of the two subsets

        self = Dictable(a = range(3))
        other = Dictable(a = range(3,6))
        c = self.merge(other, on_left = [])
        """
        other = type(self)(other)
        on_left, on_right = self._on_left_and_on_right(other, on_left, on_right)
        if len(on_left) == 0:
            pair = Dictable(idx = [np.arange(len(self) + len(other))], lhs_idx = [np.arange(len(self))], rhs_idx = [np.arange(len(other))], lhs_len = len(self), rhs_len = len(other))
        else:
            pair = self.pair(other, on_left, on_right)
        return self._join(pair, other, on_left, on_right, merge)
    
    def __mul__(self, other):
        return self.merge(other)
        
    def xor(self, other, on_left=None, on_right=None):
        """
        xor is an extremely useful function as, unlike left join, it tells us which original records we have not been able to match in other
        >>> students = Dictable(name = ['Adam', 'Beth', 'Eve'])
        >>> lunch = Dictable(name = ['Adam','Eve'], lunch = ['Bread', 'Apple'])
        >>> students_who_didnt_eat = students.xor(lunch)
        >>> assert eq(students_who_didnt_eat, Dictable(name = 'Beth'))
        """
        other = type(self)(other)
        on_left, on_right = self._on_left_and_on_right(other, on_left, on_right)
        if not on_left and not on_right:
            return self
        pair = self.pair(other, on_left, on_right)
        return self._left_xor(pair, other)

    def __truediv__(self, other):
        return self.xor(other)

    def right_xor(self, other, on_left=None, on_right=None):
        other = type(self)(other)
        on_left, on_right = self._on_left_and_on_right(other, on_left, on_right)
        if not on_left and not on_right:
            return other
        pair = self.pair(other, on_left, on_right)
        return self._right_xor(pair, other)

    def left_join(self, other, on_left=None, on_right=None):
        other = type(self)(other)
        pair = self.pair(other, on_left, on_right)
        return self._join(pair, other) + self._left_xor(pair, other)

    def right_join(self, other, on_left=None, on_right=None):
        other = type(self)(other)
        pair = self.pair(other, on_left, on_right)
        return self._join(pair, other) + self._right_xor(pair, other)
        
    def to_tree(self, pattern, tree = dict):
        """
        self = Dictable(name = ['alan', 'beth', 'charles'], surname = ['smith', 'jones', 'patel'], gender = ['m','f','m'])
        t1 = self.to_tree(pattern = 'students/%name/%surname')
        t2 = self['students/%name/%surname']
        assert t1 == {'students': {'alan': 'smith', 'beth': 'jones', 'charles': 'patel'}}
        assert t1 == t2
        """
        pattern = _as_pattern(pattern)
        items = [_pattern_to_item(pattern, row) for row in self]
        return items_to_tree(items = items, tree = tree)

