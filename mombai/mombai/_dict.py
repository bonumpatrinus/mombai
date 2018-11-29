from _collections_abc import dict_keys
from mombai._containers import slist, as_list, args_zip, args_to_list, eq
from mombai._decorators import getargs, try_list
from copy import deepcopy

def _relabel(key, relabels):
    """
    >>> relabels = dict(b='BB', c = lambda value: value.upper())
    >>> assert _relabel('a', relabels) == 'a'
    >>> assert _relabel('b', relabels) == 'BB'
    >>> assert _relabel('c', relabels) == 'C'
    """
    if key not in relabels:
        return key
    res = relabels[key]
    if not isinstance(res, str) and callable(res):
        return res(key)
    else:
        return res


class Dict(dict):
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
    d.map(function, **redirects) # equivalent to d[function] but allows parameter relabeling 
    d.do(functions, *keys, **redirects) # applies a sequence of function on multiple keys, each time mapping on the same original key
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
        return list(self.keys()) + super(Dict, self).__dir__()
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __delattr__(self, attr):
        del self[attr]
    def __getitem__(self, value):
        if isinstance(value, tuple):
            return [self[v] for v in value]
        elif isinstance(value, (list, dict_keys)):
            return type(self)({v: self[v] for v in value})
        elif callable(value):
            return self.map(value)
        return super(Dict, self).__getitem__(value)
    def keys(self):
        return slist(super(Dict, self).keys())
    def values(self):
        return list(super(Dict, self).values())
    def copy(self):
        return type(self)(self)
    def __call__(self, **functions):
        """
        d = Dict()(a = 1, b = 2, c = lambda a, b: a+b)
        """
        res = self.copy()
        for key, value in functions.items():
            res[key] = res[value] if callable(value) else value
        return res

    def _precall(self, function):
        """
        This is a placeholder, allowing classes that inherit to apply vectorization/parallelization of the call method
        """
        return function    

    def map(self, function, **relabels):
        """
        >>> d = Dict(a=1)
        >>> function = lambda x: x+2
        >>> assert d.map(function, a='x') == 3
        """
        args = getargs(function)
        relabels = relabels.get('relabels', relabels)
        parameters = {_relabel(key, relabels): value for key, value in self.items() if _relabel(key, relabels) in args}
        return self._precall(function)(**parameters)

    def do(self, functions=None, *keys, **relabels):
        """
        Many times we want to apply a collection of function on multiple keys. e.g.
        we read some data as text and and then parse and finally, just the year:

        d['issue_date'] = parse_as_date(d['issue_date'])
        d['maturity'] = parse_as_date(d['maturity'])
        d['issue_date'] = to_year(d['issue_date'])
        d['maturity'] = to_year(d['maturity'])

        instead you can:
        d = d.do([parse_as_date, to_year], 'maturity', 'issue_date')
        
        >>> d = Dict(a=1, b=2, c=3)
        >>> d = d.do(str, 'a', 'b', 'c')
        >>> assert d == Dict(a = '1', b='2', c='3')
        
        >>> d = d.do(lambda value, a: value+a, ['b', 'c'])
        >>> assert d == Dict(a = '1', b='21', c='31')        
        >>> assert d.do([int, lambda value, a: value-int(a)], 'b','c') == Dict(a = '1', b=20, c=30)
        
        """
        res = self.copy()
        keys = slist(args_to_list(keys))
        relabels = relabels.get('relabels', relabels)
        for function in as_list(functions):
            args = try_list(getargs)(function, 1) 
            if keys & args:
                raise ValueError('cannot apply args both in the function args and in the keys that need to be done as result is then order-of-operation sensitive %s'%(keys & args))
            func = res._precall(function)
            for key in keys:
                pass
                parameters = {_relabel(key, relabels): value for key, value in res.items() if _relabel(key, relabels) in args}
                res[key] = func(res[key], **parameters)
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
        if not relabels: 
            return self
        return type(self)({_relabel(key, relabels) : value for key, value in self.items()})  

    def __deepcopy__(self, *args, **kwargs):
        return type(self)({key : deepcopy(value) for key, value in self.items()})

    def __eq__(self, other):
        return eq(self, other)