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

def _prehash(value):
    """
    make list and dicts more hashable
    """
    if isinstance(value, list):
        return tuple(value)
    elif isinstance(value, dict):
        return tuple(sorted(value.items()))
    else:
        return value

def Hash(value):
    """
    _hash of _hash is the same due to integers not being hashable
    _hash of dict and list are implemented
    """
    value = _prehash(value)
    if isinstance(value, (int, np.int64, np.int32)):
        return value
    else:
        return hash(value)


def _cache_key(function):
    args = getargs(function)
    if args and args[0] in ('self', 'cls'):
        def _key(*args, **kwargs):
            return (args[1:], tuple(kwargs.items()))
    else:
        def _key(*args, **kwargs):
            return (args, tuple(kwargs.items()))
    return _key


def cache_method(function):
    """
    >>> import datetime
    >>> import time
    >>> class Clock(object):
        @cache_method
        def time(self):
            return datetime.datetime.now()
    >>> c = Clock()
    >>> t = c.time()
    >>> time.sleep(1)
    >>> assert c.time() == t

    >>> d = Clock()
    >>> assert d.time()>t    
    """
    name = function.__name__
    def _key(*args, **kwargs):
        return (name, args[1:], tuple(kwargs.items()))
    def wrapped(*args, **kwargs):
        key = _prehash(_key(*args, **kwargs))
        me = args[0]
        if _hashable(key):
            me.cache = getattr(me, 'cache', {})
            if key not in me.cache:
                me.cache[key] = function(*args, **kwargs)
            return me.cache[key]
        else:
            return function(*args, **kwargs)
    result = decorate(wrapped, function)
    return result

def cache_func(function):
    def _key(*args, **kwargs):
        return (args, tuple(kwargs.items()))
    def wrapped(*args, **kwargs):
        key = _prehash(_key(*args, **kwargs))
        if _hashable(key):
            wrapped.cache =  getattr(wrapped, 'cache', {})
            if key not in wrapped.cache:
                wrapped.cache[key] = function(*args, **kwargs)
            return wrapped.cache[key]
        else: 
            return function(*args, **kwargs)
    result = decorate(wrapped, function)
    result.cache = {}
    return result
    

def cache(function):
    """
    A decorator for a function, where a function call arguments are hashable, will cache on that key

    """
    args = getargs(function)
    if args and args[0] in ('self', 'cls'):
        return cache_method(function)
    else:
        return cache_func(function)
    

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
    
    As a special case. If relabels is a string, we treat it as a function that replaces 'label' with that string:
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
    >>> assert relabeled_func(a=1,BB=2,C=3,other_parameters_that_are_now_ignaored = 100)==6
    
    Now, let us examine where the function already supports kwargs, we add relabeling for the args only!
    >>> function = lambda x, **kwargs : ' '.join(['x'*x] + [key*value for key, value in kwargs.items()]) 
    >>> assert support_kwargs()(function)(x=1, b=2, c=3) == 'x bb ccc'
    >>> assert support_kwargs(dict(x='a'))(function)(a=1, b=2, c=3) == 'x bb ccc'
    >>> assert support_kwargs(dict(x='a', y='b'))(function)(x=1, b=2, c=3) == 'x bb ccc' # we never relabel kwargs. so 'b' is passed to the function rather than 'y'        
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
            """
            We need to supprt the case where varkw isnt None: the function expect parameters whose name we do not know.
            They may still have relabels
            >>> function = lambda x, **kwargs : ''.join(['x'*x] + [key*value for key, value in kwargs.items()]) 
            >>> assert support_kwargs()(function)(x=1, b=2, c=3) == 'xbbccc'
            >>> assert support_kwargs(dict(x='a'))(function)(a = 1, b=2, c=3) == 'xbbccc'
            support_kwargs(dict(x='a', y='b'))(function)(x=1, b=2, c=3) == 'xbbccc' # we never relabel kwargs. 
            """
            args2keys = {arg : relabel(arg, relabels) for arg in argspec.args}
            parameters = {arg : kwargs.pop(key) for arg, key in args2keys.items() if key in kwargs}
            if argspec.varkw is not None:
                parameters.update(kwargs)
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
        print(' '.join([str(t0), function.__name__] + [_txt(a) for a in args] + ['%s=%s'%(key, _txt(a)) for key, a in kwargs.items()]))
        res = function(*args, **kwargs)
        return res
    return decorate(wrapped, function)

def dict_loop(function):
    """
    wraps a function to loop over the first argument if it is a of a type dict
    """
    argspec = getargspec(function)
    top = argspec.args[0] if argspec.args else ''
    def wrapped(*args, **kwargs):
        if len(args): 
            if isinstance(args[0], dict):
                def _k(value, key):
                    return value[key] if isinstance(value, dict) and value.keys() == args[0].keys() else value
                return type(args[0])({key : wrapped(*(_k(a, key) for a in args), **{k:_k(v, key) for k,v in kwargs.items()}) for key in args[0].keys()})
            else:
                return function(*args, **kwargs)
        elif len(kwargs) and top in kwargs and isinstance(kwargs[top], dict):
            def _k(value, key):
                return value[key] if isinstance(value, dict) and value.keys() == kwargs[top].keys() else value
            return type(kwargs[top])({key : wrapped(*(_k(a, key) for a in args), **{k:_k(v, key) for k,v in kwargs.items()}) for key in kwargs[top].keys()})
        else:
            return function(*args, **kwargs)
    return decorate(wrapped, function)


def array_loop(tp = list):
    """
    returns a decorator that wraps a function to loop over the first argument if it is a of a type tp
    """
    def looper(function):
        argspec = getargspec(function)
        top = argspec.args[0] if argspec.args else ''
        def wrapped(*args, **kwargs):
            if len(args): 
                if isinstance(args[0], tp):
                    def _k(value, key):
                        return value[key] if isinstance(value, tp) and len(value) == len(args[0]) else value
                    return type(args[0])([wrapped(*(_k(a, key) for a in args), **{k:_k(v, key) for k,v in kwargs.items()}) for key in range(len(args[0]))])
                else:
                    return function(*args, **kwargs)
            elif len(kwargs) and top in kwargs and isinstance(kwargs[top], tp):
                def _k(value, key):
                    return value[key] if isinstance(value, tp) and len(value) == len(kwargs[top]) else value
                return type(kwargs[top])([wrapped(*(_k(a, key) for a in args), **{k:_k(v, key) for k,v in kwargs.items()}) for key in range(len(kwargs[top]))])
            else:
                return function(*args, **kwargs)
        return decorate(wrapped, function)
    return looper

list_loop = array_loop(list)

def _getattr(obj, attr):
    """
    A simple extension of getattr(obj, attr) to allow us to _getattr(obj, 'call') which is slightly prettier
    """
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else:
        objattrs = [a for a in dir(obj) if a.replace('_', '') == attr]
        if len(objattrs) == 1:
            return getattr(obj, objattrs[0])
    raise AttributeError('Attribute %s not found in %s'%(attr, obj))


def callattr(obj, attr='call', *args, **kwargs):
    """
    small wrappper to allow us to implement calling of an object attributes
    """
    return _getattr(obj, attr)(*args, **kwargs)

def callitem(obj, item, *args, **kwargs):
    """
    small wrappper to allow us to implement calling of an object attributes
    """
    return obj[item](*args, **kwargs)
