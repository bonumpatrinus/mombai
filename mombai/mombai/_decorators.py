import inspect
from functools import partial
import sys
from numpy import nan
import collections
import logging
import datetime
import numpy as np

version = sys.version_info
if version.major < 3:
    ARGSPEC = 'argspec' 
else:
    ARGSPEC  = 'fullargspec'

def argspec_update(argspec, **kwargs):
    """
    Allows us to copy an existing argspec, updating specific parameters specified in kwargs
    """
    tp = type(argspec)
    params = {key : getattr(argspec, key) for key in dir(tp) if not key.startswith('_') and key not in ('count','index')}
    params.update(kwargs)
    return tp(**params)

  
def getargspec(function):
    if hasattr(function, ARGSPEC):
        return getattr(function, ARGSPEC)
    elif isinstance(function, partial):
        return getargspec(function.func)
    return getattr(inspect, 'get%s'%ARGSPEC)(function)
    
def getargs(function, n = 0):
    """
    get the name of the args after allowing for the first n args to be provided as *args by the user
    """
    argspec = getargspec(function)
    if argspec.varargs or n==0:
        return argspec.args
    else:
        return argspec.args[n:]

def decorate(wrapped, function):
    setattr(wrapped, ARGSPEC, getargspec(function))
    for attr in ['__name__', '__doc__']:
        if hasattr(function, attr):
            setattr(wrapped, attr, getattr(function, attr))
    return wrapped


def _hashable(key):
    """
    >>> key = ([1,2],) ## A list hiding inside a tuple
    >>> assert isinstance(key, collections.Hashable)
    >>> assert not _hashable(key)
    """
    return min([_hashable(k) for k in key]) if isinstance(key, tuple) and len(key) else isinstance(key, collections.Hashable)
    

def _cache_key(function):
    args = getargs(function)
    if args and args[0] in ('self', 'cls'):
        def _key(*args, **kwargs):
            return (args[1:], tuple(kwargs.items()))
    else:
        def _key(*args, **kwargs):
            return (args, tuple(kwargs.items()))
    return _key


def cache(function):
    """
    A decorator for a function, where a function call arguments are hashable, will cache on that key
    """
    _key = _cache_key(function)
    def wrapped(*args, **kwargs):
        key = _key(*args, **kwargs)
        if _hashable(key):
            if key not in wrapped.cache:
                wrapped.cache[key] = function(*args, **kwargs)
            return wrapped.cache[key]
        else: 
            return function(*args, **kwargs)
    result = decorate(wrapped, function)
    result.cache = {}
    return result

def try_value(value):
    def decorator(function):
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                return value
        return decorate(wrapped, function)
    return decorator


try_nan = try_value(nan)
try_none = try_value(None)
try_zero = try_value(0)
try_str = try_value('')
try_list = try_value([])
try_dict = try_value({})
try_false = try_value(False)
try_true = try_value(True)

def try_back(function):
    """
    tries to calculate a function. on failing, returns its first argument
    """
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            return args[0] if args else kwargs[getargs(function)[0]]
    return decorate(wrapped, function)

def relabel(key, relabels=None):
    """
    relabel does quite a few things
    
    Let us first look at how we rename strings:
    >>> relabels = dict(b='BB', c = lambda value: value.upper())
    >>> assert relabel('a', relabels) == 'a'
    >>> assert relabel('b', relabels) == 'BB'
    >>> assert relabel('c', relabels) == 'C'

    Now we can use relabels to relabel a dict/list/tuple as well
    >>> assert relabel(dict(a = 1, b=2, c=3), relabels) == {'a': 1, 'BB': 2, 'C': 3}
    >>> assert relabel(['a', 'b', 'c'], relabels) == ['a', 'BB', 'C']
    >>> assert relabel(('a', 'b', 'c'), relabels) == ('a', 'BB', 'C')
    >>> assert relabel(None, relabels) is None
    
    Now, relabels can be a function rather than just a dict: 
    >>> assert relabel(dict(a = 1, b=2, c=3), lambda label: label*2) == {'aa': 1, 'bb': 2, 'cc': 3}
    
    Finally, a special case. If relabels is a string, we treat it as a function that replaces 'label' with that string:
    >>> assert relabel('label_in_text_is_renamed', 'market') == 'market_in_text_is_renamed'

    Lastly, if key is a FUNCTION. Then we essentially use the relabels to change the function signature.
    >>> func = lambda a, b, c : a+b+c
    >>> assert getargs(func) == ['a', 'b', 'c']
    >>> relabeled_func = relabel(func, relabels)
    >>> assert getargspec(relabeled_func).args == ['a', 'BB', 'C']
    >>> assert relabeled_func(1,2,3) == 6
    >>> assert relabeled_func(a=1,BB=2,C=3)==6
    """
    if isinstance(key, dict):
        return type(key)({relabel(k, relabels) : v for k, v in key.items()})
    elif isinstance(key, (list, tuple)):
        return type(key)([relabel(k, relabels) for k in key])
    elif key is None:
        return key
    elif callable(key):
        return support_kwargs(relabels)(key)
    elif relabels is None:
        return key
    if isinstance(relabels, str):
        return key.replace('label', relabels)
    if callable(relabels):
        return relabels(key)
    if key not in relabels:
        return key
    res = relabels[key]
    if not isinstance(res, str) and callable(res):
        return res(key)
    else:
        return res
    

def support_kwargs(relabels=None):
    """
    convert a function to support kwargs. If a function has no argspec, will just feed the un-named parameters
    if relabels are provided, the relabels are used to change the function signature!
    >>> relabels = dict(b='BB', c = lambda value: value.upper())
    >>> func = lambda a, b, c : a+b+c
    >>> assert getargs(func) == ['a', 'b', 'c']
    >>> relabeled_func = support_kwargs(relabels)(func)
    >>> assert getargspec(relabeled_func).args == ['a', 'BB', 'C']
    >>> assert relabeled_func(1,2,3) == 6
    >>> assert relabeled_func(a=1,BB=2,C=3)==6
    """
    relabels = relabels  or {}
    def decorator(function):
        try:
            argspec = getargspec(function)
        except TypeError:
            def wrapped(*args, **kwargs):
                return function(*args)
            return wrapped  
        def wrapped(*args, **kwargs):
            args2keys = {arg : relabel(arg, relabels) for arg in argspec.args}
            parameters = {arg : kwargs[key] for arg, key in args2keys.items() if key in kwargs}
            return function(*args, **parameters)
        wrapped = decorate(wrapped, function)
        setattr(wrapped, ARGSPEC, argspec_update(argspec, varkw = 'kwargs', args = relabel(argspec.args, relabels), defaults = relabel(argspec.defaults, relabels)))
        return wrapped
    return decorator

def _txt(value):
    return str(value) if isinstance(value, (int, str, float, bool, datetime.datetime)) else str(type(value))

def profile(function):
    def wrapped(*args, **kwargs):
        t0 = datetime.datetime.now()
        res = function(*args, **kwargs)
        t1 = datetime.datetime.now()
        print(' '.join([str(t1-t0), function.__name__] + [_txt(a) for a in args] + ['%s=%s'%(key, _txt(a)) for key, a in kwargs.items()]))
        return res
    return decorate(wrapped, function)