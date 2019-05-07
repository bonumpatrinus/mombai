from _collections_abc import dict_keys
from mombai._compare import eq
from mombai._containers import slist, as_list, args_zip, args_to_list
from mombai._decorators import getargs, try_list, decorate, support_kwargs, relabel
from mombai._dict_utils import pass_thru, first, last
from functools import partial
from copy import deepcopy

class Dictattr(dict):
    """
    Dictattr is our base dict and inherits from dict with support to attribute access
    >>> a = Dictattr(a = 1, b = 2)
    >>> assert a['a'] == a.a
    >>> a.c = 3
    >>> assert a['c'] == 3
    >>> del a.c
    >>> assert list(a.keys()) == ['a','b']
    >>> assert a['a','b'] == [1,2]
    >>> assert a[['a','b']] == Dictattr(a = 1, b=2)
    >>> assert not a == dict(a=1,b=2)
    
    >>> a = Dictattr(a=1)
    >>> hasattr(a, 'tes')
    """
    def __sub__(self, other):
        return type(self)({key: value for key, value in self.items() if  key in self.keys() - other})
    def __and__(self, other):
        return type(self)({key: value for key, value in self.items() if  key in self.keys() & other})
    def __add__(self, other):
        res = self.copy()
        res.update(other)
        return res
    def __dir__(self):
        return list(self.keys()) + super(Dictattr, self).__dir__()
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(str(e))
        return super(Dictattr, self).__getattr__(attr) if attr.startswith('_') else self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError as e:
            raise AttributeError(str(e))
    def __getitem__(self, value):
        if isinstance(value, tuple):
            return [self[v] for v in value]
        elif isinstance(value, (list, dict_keys)):
            return type(self)({v: self[v] for v in value})
        return super(Dictattr, self).__getitem__(value)
    def keys(self):
        return slist(super(Dictattr, self).keys())
    def values(self):
        return list(super(Dictattr, self).values())
    def copy(self):
        return type(self)(self)
    def __deepcopy__(self, *args, **kwargs):
        return type(self)({key : deepcopy(value) for key, value in self.items()})
    def __eq__(self, other):
        return eq(self, other)
    def __getstate__(self):
        return dict(self)
    def __setstate__(self, d):
        self.update(d)
    def __dict__(self):
        return dict(self)

         
def _where(cond, key, value):
    """
    condition can be both a function or a constant value
    value can be both a function or a constant value
    """
    condition = relabel(cond if callable(cond)  else partial(eq , y = cond))
    function = relabel(value if callable(value) else lambda *args, **kwargs: value)
    def wrapped(*args, **kwargs):
        kwargs_ = {k:v for k,v in kwargs.items() if k!=key}
        if condition(*args, **kwargs_):
            return args[0]
        else:
            return function(*args, **kwargs_)
    return wrapped        

def _mask(cond, key, value):
    """
    We are getting into serious argpspec messing territory
    
    def where(self, cond, key = lambda a, b: a +b):
        self[key] = where(cond, key, function)(self[key], **self)
    """
    condition = relabel(cond if callable(cond)  else partial(eq , y = cond))
    function = relabel(value if callable(value) else lambda *args, **kwargs: value)
    def wrapped(*args, **kwargs):
        kwargs_ = {k:v for k,v in kwargs.items() if k!=key}
        if condition(*args, **kwargs_):
            return function(*args, **kwargs_)
        else:
            return args[0]
    return wrapped


class Dict(Dictattr):
    """
    Dict inherits from dict with some key additional features. 
    The aim is to transform dict into a mini "container of variables" in our research:
     
    Here is the usual pattern of our research:
        
    def function(a,b):
        c = a + b
        d = c * a
        e = d - c
        return e
    
    def other_function(a, e):
        #note that b isn't needed
        return e+a
        
    x = dict(a=1, b=2) 
    x['e'] = function(x['a'], x['b'])
    x['f'] = other_function(x['a'], x['e'])
    
    The trouble is...:
    1) during research we are not sure if the code for function is right
    2) we don't have a visibility on what happens internally
    3) if functions fails, debugging is a pain and tracking log tricky. 
    4) we cannot even feed other_function(**x) since x has b that other_function does not want
    
    so... Dict makes this easier by inspecting the function's argument names, matching its keys to the function's kwargs:
    x = Dict(a=1, b=2) 
    x['e'] = x[function]
    x['f'] = x[other_function]

    If we want to build our result gradually, then we do this:
    x['c'] = x[lambda a,b : a+b]    
    x['d'] = x[lambda a,c : a*c]    
    x['e'] = x[lambda d,c : d-c]    
    we can run each of the lines separately and examine the result as we go along.

    We can add the new variable by applying the function: 
    x = x(c = lambda a,b : a+b)    
    x = x(d = lambda a,c : a*c)
    x = x(e = lambda d,c : d-c)
    
    allowing us to "chain operations"
    x = x(c = lambda a,b : a+b)(d = lambda a,c : a*c)(e = lambda d,c : d-c) ## or even...
    x = x(c = lambda a,b : a+b, d = lambda a,c : a*c, e = lambda d,c : d-c)    
    assert x == dict(a = 1, b=2, c=3, d=3, e=0)
    
    Dict can also be used as a mini "calculation graph/network", where we define the order of calculations and then perform them later on values:
    
    calculations = Dict(c = lambda a,b : a+b, d = lambda a,c : a*c, e = lambda d,c : d-c)
    inputs = Dict(a=1, b=2)
    values = inputs(**calculations)    
    graph = inputs + calculations
    values = Dict()(**graph)
    
    In addition, there are a few features to quicken development:
    d.apply(function, **redirects) # equivalent to d[function] but allows parameter relabeling 
    d.do(functions, *keys, **redirects) # applies a sequence of function on multiple keys, each time mapping on the same original key
    """
    def __call__(self, *relabels, **functions):
        """
        The call function allows us to assign to new keys, new values:
        >>> d = Dict(a = 1)
        >>> d = d(b = 2)
        >>> assert d == Dict(a=1,b=2)
        
        
        In addition, we can add new columns using functions of existing columns:
        >>> d = d(c = lambda a, b: a+b)
        >>> assert d == Dict(a=1,b=2,c=3)
        
        Sometime, we may have an existing function that we want to use. In which case we can map the function's args to the keys:
        >>> def add(x,y):
        >>>    return x+y
        >>> d = d(dict(x='a',y='b', z='c'), z = add) # note that we are allowed to re-label the return key as well!
        >>> assert d == Dict(a=1,b=2,c=3)

        We allow us to actually loop over multiple keys and run the function twice: 
        >>> d = d(dict(x='a',y='b',z='c'), dict(x='c',y='b',z='d'), z = add)
        >>> assert d == Dict(a=1, b=2, c=3, d=5)

        The final functionality is not for the faint of hearts. Using 'label' in the args of the function is treated as a special case. Allowing us to write succinctly: 
            
        >>> d = Dict(a=1,b=2,c=3)
        >>> d = d('b', 'c', label2 = lambda label: label**2)
        >>> assert d == Dict(a=1,b=2,c=3, b2=4, c2=9)
        >>> d = d('b','c', label_cubed_plus_a = lambda label, label2, a: label2 * label+a)
        >>> assert d == Dict({'a': 1, 'b': 2, 'c': 3, 'b2': 4, 'c2': 9, 'b_cubed_plus_a': 9, 'c_cubed_plus_a': 28})

        """
        res = self.copy()
        if len(relabels) == 0:
            relabels = [pass_thru]
        for key, value in functions.items():
            if callable(value):
                for r in relabels:
                    res[relabel(key, r)] = res.apply(value, r)
            else:
                res[key] = value
        return res

    def __getitem__(self, value):
        if isinstance(value, tuple):
            return [self[v] for v in value]
        elif isinstance(value, (list, dict_keys)):
            return type(self)({v: self[v] for v in value})
        elif callable(value):
            return self.apply(value)
        return super(Dict, self).__getitem__(value)


    def _precall(self, function, relabels=None):
        """
        Make the function being able to take more that just the key words needed and relabel the internal parameters
        """
        return function

    def apply(self, function, relabels=None):
        """
        >>> self = Dict(a=1)
        >>> function = lambda x: x+2
        >>> assert self.apply(function, relabels = dict(a = 'x')) == 3
        >>> assert self.apply(function, relabels = lambda arg: arg.replace('a', 'x')) == 3
        """
        return self._precall(relabel(function, relabels))(**self)
                

    def do(self, functions=None, *keys, **relabels):
        """
        Many times we want to apply a collection of function on multiple keys, 
        returning the resulting value to the same key. E.g.
        we read some data as text and and then parse and finally, just the year:
            
        >>> from dateutil import parser
        >>> to_year = lambda date: date.year
        >>> from mombai import Dict
        
        >>> d1 = Dict(issue_date = '1st Jan 2001', maturity = '2nd Feb 2010')
        >>> d2 = Dict(issue_date = '1st Jan 2001', maturity = '2nd Feb 2010')
        
        >>> d1['issue_date'] = parser.parse(d1['issue_date'])
        >>> d1['maturity'] = parser.parse(d1['maturity'])
        >>> d1['issue_date'] = to_year(d1['issue_date'])
        >>> d1['maturity'] = to_year(d1['maturity'])
        
        #instead you can:
        >>> d2 = d2.do([parser.parse, to_year], 'maturity', 'issue_date')
        >>> assert d1 == d2
        
        
        >>> d = Dict(a=1, b=2, c=3)
        >>> d = d.do(str, 'a', 'b', 'c')
        >>> assert d == Dict(a = '1', b='2', c='3')
        >>> d = d.do(lambda value, a: value+a, ['b', 'c'])
        >>> assert d == Dict(a = '1', b='21', c='31')        
        >>> assert d.do([int, lambda value, a: value-int(a)], 'b','c') == Dict(a = '1', b=20, c=30)
        
        >>> d = Dict(a=1,b=2,c=3)
        >>> assert d.do(lambda value, other: value+other, 'b','c', other = 'a') == Dict(a=1,b=3,c=4)
        >>> assert d.do(lambda value, other: value+other, 'b','c', relabels = lambda arg: arg.replace('other', 'a')) == Dict(a=1,b=3,c=4)
        """
        res = self.copy()
        keys = slist(args_to_list(keys))
        relabels = relabels.get('relabels', relabels)
        for function in as_list(functions):
            func = res._precall(relabel(function, relabels))
            for key in keys:
                res[key] = func(res[key], **(res-key))
        return res

    def where(self, cond, *function, **functions):
        """
        where and mask implement a similar interface to pd.DataFrame
        :cond is checked against all keys that are in functions. This can be a function or just a simple constant value
        :function is a single value function that is applied uniformly to all keys
        :functions is a dict of values to which the keys are changed if the coditioned isn't matched

        >>> d = Dict(a = None, b=None, c='1')
        >>> x = d.where('not true', a=0, b=1, c=2)
        >>> assert x == Dict(a=0, b=1, c=2) ## condition is False always, so all values are changed
        >>> assert d.where(lambda value: value is None, a=0, b=1, c=2) == Dict(a=None, b=None, c=2) ## only c is changed
        >>> assert d.where(None, a = 0, b=1, c=2) == Dict(a=None, b=None, c=2) ## Can be done using a simple equality
        >>> assert d.where(None, a=float, b=float, c=float) == Dict(a=None, b=None, c=0.0) ## converts to float values that are not None
        >>> assert d.where(None, float) == Dict(a=None, b=None, c=0.0) ## converts to float values that are not None
        
        >>> d = Dict(a = None, b=None, c='1')
        >>> x = d.where(lambda value: value is not None, a=0, b=0, c=0)
        >>> assert x == Dict(a=0, b=1, c='1')
        
        :using a condition that is a type
        >>> d = Dict(a = '1', b = 2, c = 3.0)
        >>> assert d.mask(str, a = lambda a: float(a)) == Dict(a = 1.0, b = 2.0, c=3.0) ## what isn't float is converted to float        
        """
        res = self.copy()
        mappers = {key : function[0] for key in self} if len(function) ==1 else {}
        mappers.update(functions)
        for key, value in mappers.items():
            res[key]= self._precall(_where(cond, key, value))(res[key], **res)
        return res

    def mask(self, cond, *function, **functions):
        """
        where and mask implement a similar interface to pd.DataFrame
        :cond is checked against all keys that are in functions. This can be a function or just a simple constant value
        :function is a single value function that is applied uniformly to all keys
        :functions is a dict of values to which the keys are changed if the coditioned isn't matched

        >>> d = Dict(a = None, b=None, c='1')
        >>> assert d.mask('not true', a=0, b=1, c=2) == d ## condition is False always, so all values are unchanged
        >>> assert d.mask(False, a=0, b=1, c=2) == d ## condition is False always, so all values are unchanged
        >>> assert d.mask(None, a=0, b=1, c=2) == Dict(a=0, b=1, c='1') ## only c is changed
        >>> assert d.mask(None, 0) == Dict(a=0, b=0, c='1') ## Can be done using a simple equality
        >>> assert d.mask(None, 0, b=1) == Dict(a=0, b=1, c='1') ## every None goes to 0 except b
        >>> assert d.mask(lambda value: value is not None, float) == Dict(a=None, b=None, c=0.0) ## converts to float values that are not None
        >>> assert d.where(None, float) == Dict(a=None, b=None, c=0.0) ## converts to float values that are not None
        
        >>> d = Dict(a = None, b=None, c='1')
        >>> x = d.where(lambda value: value is not None, a=0, b=0, c=0)
        >>> assert x == Dict(a=0, b=1, c='1')
        
        :using a condition that is a type
        >>> d = Dict(a = '1', b = 2, c = 3.0)
        >>> assert d.mask(str, a = lambda a: float(a)) == Dict(a = 1.0, b = 2.0, c=3.0) ## what isn't float is converted to float        
        """
        res = self.copy()
        mappers = {key : function[0] for key in self} if len(function) ==1 else {}
        mappers.update(functions)
        for key, value in mappers.items():
            res[key] = res._precall(_mask(cond, key, value))(res[key], **res)
        return res

    def relabel(self, **relabels):
        """quick functionality to relabel the keys
        if existing key is not in the relabels, it stays the same
        if the relabel is another simple column name, it gets used
        if the relabel is a function, use this on the key.
        
        >>> d = Dict(a=1, b=2, c=3)
        >>> assert d.relabel(b = 'bb', c=lambda value: value.upper()) == Dict(a = 1, bb=2, C=3)
        >>> assert d.relabel(relabels = dict(b='bb', c=lambda value: value.upper())) == Dict(a=1, bb=2, C=3)
        """
        relabels = relabels.get('relabels', relabels)
        return relabel(self, relabels)
