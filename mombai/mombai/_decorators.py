import inspect
from functools import partial
import sys
from numpy import nan
import collections

version = sys.version_info
ARGSPEC = 'argspec' if version.major<3 else 'fullargspec'
    
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
    wrapped.argspec =  getargspec(function)
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
    decorator.__doc__ = "tries to evaluate a function. if fails, returns %s"%(value)
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
