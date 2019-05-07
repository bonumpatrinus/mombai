import numpy as np
import jsonpickle as jp
import datetime
from mombai._containers import as_list, is_array, many2one
from mombai._decorators import Hash, getargs, getargspec, dict_loop, try_none
from mombai._dates import dt, fraction_of_day, seconds_of_day, timedelta_of_day
from mombai._dict import Dict
from mombai._dict_utils import items_to_tree, tree_items
from functools import partial
from copy import copy
from inspect import FullArgSpec


def _per_cell(value, f):
    if is_array(value):
        return type(value)([_per_cell(v, f) for v in value])
    elif isinstance(value, Cell):
        return f(value)
    else:
        return value

@try_none
def _getargspec(function):
    res = getargspec(function)
    if res.args and res.args[0] in ('cls','self'):
        res = FullArgSpec(args=res.args[1:], varargs=res.varargs, varkw=res.varkw, defaults=res.defaults, kwonlyargs=res.kwonlyargs, kwonlydefaults=res.kwonlydefaults, annotations=res.annotations)
    return res
        
_to_id = partial(_per_cell, f = lambda v: '@%s'%(v.id if isinstance(v.node, dict) else v.node))
_to_id.__doc__ = "Converts a cell into its id. If it was a dict, then an v.id, its hash is used. Otherwise, if it is a single value, use that value as reference"
_dict_to_node = lambda d: None if d is None or len(d) == 0 else d['node'] if len(d)==1 and 'node' in d else d
_node_to_dict = lambda n: Dict({} if n is None else n if isinstance(n, dict) else {'node' : n})
_call = partial(_per_cell, f = lambda v: v())


def _as_asof(asof):
    """
    A timestamp function to decide a cutoff of a time after which we recalculate
    so, if we set a cell to update with an asof of 60, then 
    >>> now = dt.now()
    >>> minute_ago = _as_asof(60)
    >>> assert (now - minute_ago).seconds <=60
    """
    if asof is None:
        return datetime.datetime.now()
    if isinstance(asof, int):
        return datetime.datetime.now() - datetime.timedelta(seconds = asof)
    if isinstance(asof, datetime.timedelta):
        return datetime.datetime.now() - asof
    return dt(asof)


def passthru(x):
    return x


def _args_that_are_kwargs(argspec, args, kwargs):
    varargs = len(args) if argspec.varargs is None and args else 0
    if varargs:
        duplicates = set(argspec.args[:varargs]) & set(kwargs.keys())
        if len(duplicates):
            raise TypeError('Got multiple values for argument %s'%duplicates)
    return varargs

def _as_fak_strict(function, args, kwargs):
    if not is_array(args):
        raise TypeError('args must be an array %s, perhaps initiate Cell using Cell.f?'%args)
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be a dict %s, perhaps initiate Cell using Cell.f?'%kwargs)
    if isinstance(function, Cell):
        if len(args)==0 and len(kwargs)==0:
            args = function.args
            kwargs = function.kwargs
        function = function.function
    if callable(function):
        argspec = _getargspec(function)
        if argspec is not None:
            varargs = _args_that_are_kwargs(argspec, args, kwargs)
            if varargs:
                duplicates = set(argspec.args[:varargs]) & set(kwargs.keys())
                if len(duplicates):
                    raise TypeError('Got multiple values for argument %s'%duplicates)
            nondefault = argspec.args[varargs :-len(argspec.defaults)] if argspec.defaults else argspec.args[varargs:]
            missing  = set(nondefault) - set(kwargs.keys())
            if len(missing):
                raise TypeError('Missing variables %s as positional arguments please use F.at() if you want these defaulting to reference.'%missing)
            unrecognised = set(kwargs.keys()) - set(argspec.args)
            if len(unrecognised):
                raise TypeError('Unexpected keword argument %s'%unrecognised)
    return function, args, kwargs

def _as_fak_relaxed(function, args, kwargs):
    if not is_array(args):
        raise TypeError('args must be an array %s'%args)
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be a dict %s'%kwargs)
    if isinstance(function, Cell):
        if len(args)==0 and len(kwargs)==0:
            args = function.args
            kwargs = function.kwargs
        function = function.function
    if callable(function):
        argspec = _getargspec(function)
        if argspec is not None:
            varargs = _args_that_are_kwargs(argspec, args, kwargs)
            nondefault = argspec.args[varargs :-len(argspec.defaults)] if argspec.defaults else argspec.args[varargs:]
            kwargs_ = {key : '@'+key for key in nondefault}
            kwargs_.update(kwargs)
            kwargs = {key : value for key, value in kwargs_.items() if key in argspec.args}
    return function, args, kwargs

class Cell(object):
    """
    All cells data structure will be similar:
        
    :node   An id, but can be an entire dict of metadata. We use node to identify the cell uniquely
    :config Some config used for the property of the cell (rather than for the evaluation of the function)
    
    Parameters used to value the cell:
    :function
    :args
    :kwargs
    
    We provide a two-stage construction 
    :'__init__' and 'at', both construct just the function(*args, **kwargs) bit
    :'new'
    """
    def __init__(self, function, args=(), kwargs={}, node=None, config={}):
        """
        Construction
        >>>
        >>> x = Cell(1)
        >>> assert x() == 1 and x.function == passthru
        >>> x = F(1); y = F(passthru, 1); z = F(passthru, x = 1)
        >>> for f in [x,y,z]:
        >>>     assert f.function == passthru and f.kwargs == dict(x = 1)
        >>> import pytest
        >>> with pytest.raises(TypeError):
        >>>     F(lambda x: x)
        >>> with pytest.raises(TypeError):
        >>> F(lambda x: x, y = 'too many', x = 'provided')
        >>> with pytest.raises(TypeError):
        >>>     F(lambda x: x, 'x as arg', x = 'as kwarg')
        """
        f, a, k =  _as_fak_strict(function, args, kwargs)
        self.function = f
        self.args = a
        self.kwargs = k
        self.node = function.node if node is None and isinstance(function, Cell) else node
        self.config = Dict(function.config if len(config) == 0 and isinstance(function, Cell) else config)
        
    @classmethod
    def f(cls, function, *args, **kwargs):
        f,a,k = _as_fak_strict(function, args, kwargs)
        return cls(f,a,k)

    @classmethod
    def at(cls, function, *args, **kwargs):
        """
        constructor where missing variables default to @variable to be determined later
        """
        f,a,k = _as_fak_relaxed(function, args, kwargs)
        return cls(f,a,k)

    @classmethod
    def cfg(cls, function, node = None,  **config):
        """
        constructor with config and node being the important variables
        """
        f,a,k = _as_fak_relaxed(function, (), {})
        return cls(f,a,k,node,config)

    def config_update(self, **cfg):
        """
        updates the config with cfg
        """
        res = copy(self)
        res.config = items_to_tree(tree_items(cfg), res.config)
        return res
    
    def reconfig(self, **cfg):
        """
        updates the config with cfg
        """
        res = copy(self)
        res.config = Dict(cfg)
        return res
    

    @property
    def inputs(self):
        return [self.function] + list(self.args) + list(self.kwargs.values())
    
    def _apply(self, call, *args, **kwargs):
        f = call(self.function, *args, **kwargs)
        a = tuple(call(arg, *args, **kwargs) for arg in self.args)
        k = {key: call(value, *args, **kwargs) for key, value in self.kwargs.items()}
        return f, a, k
    
    def apply(self, call, *args, **kwargs):
        res = copy(self)
        f,a,k = self._apply(call, *args, **kwargs)
        res.function = f
        res.args = a
        res.kwargs = k
        return res

    def __call__(self):
        f,a,k = self._apply(_call)
        if not callable(f):
            if len(a) or len(k):
                raise TypeError('function is not callable')
            else:
                return f
        else:
            return f(*a, **k)

    def __eq__(self, other):
        return type(other) == type(self) and self.node == other.node and self.function == other.function and self.args == other.args and self.kwargs == other.kwargs and self.config == other.config 

    def __repr__(self):
        res = getattr(self.function, '__name__', 'function') + '(' + (str(self.args)[1:-1] if self.args else '') + (' **%s'%self.kwargs if self.kwargs else '') + ')'
        return '@%s: %s'%(self.id, res) if self.node else res

    def update(self):
        """ checks if a cell needs updating By default, unless a Cell has been declared to be cached, we assume it is volatile and return True """
        return True

    @property
    def id(self):
        """
        A hash of self.node        
        """
        return str(Hash(self.node))
    
    def metadata(self):
        res = Dict()
        res['id'] = self.id
        res['node'] = self.node
        res['json']  = jp.encode(self)
        res['type'] = jp.encode(type(self))
        res['function'] = jp.encode(self.function)
        res['config'] = self.config
        return res
    
    def to_id(self):
        f, a, k = self._apply(_to_id)
        return type(self)(f, a, k, node = self.node, config = self.config)

    def __add__(self, rhs):
        """
        We apply Dict like behviour of updating to the node identity.
        """
        res = copy(self)
        res.node = _dict_to_node(_node_to_dict(self.node) + rhs)
        return res

    def __sub__(self, rhs):
        """
        We apply Dict like behviour of updating to the node indentity.
        """
        res = copy(self)
        res.node = _dict_to_node(_node_to_dict(self.node) - rhs)
        return res

    def relabel(self, *args, **kwargs):
        """
        We apply Dict like behviour of updating to the node in the cell.
        """
        res = copy(self)
        res.node = _dict_to_node(_node_to_dict(self.node).relabel(*args, **kwargs))
        return res


_maxbool = many2one(False)(max)

def _update(arg):
    """
    returns the max of all cell.update() in an array
    """
    if isinstance(arg, Cell):
        return arg.update()
    elif is_array(arg):
        for a in arg:
            if _update(a):
                return True
    return False        
        

class MemCell(Cell):
    """
    In-Memory caching Cell
    """

    def __init__(self, function, args=(), kwargs={}, node=None, config={}):
        super(MemCell, self).__init__(function, args, kwargs, node, config)
        self.cache = self._load_cache()
        
#   f = Cell.f
#    @classmethod
#    def f(cls, function, *args, **kwargs):
#        f,a,k = _as_fak_strict(function, args, kwargs)
#        return cls(f,a,k)


    def _load_cache(self):
        """ for a file-based or db-based cache, we can implement this"""
        return Dict()

    def _to_cache(self, cache):
        """ for a file-based or db-based cache, we can implement this"""
        return cache
    
    def update(self):
        return len(self.cache) == 0 or _update(self.function) or _update(self.args) or _update(list(self.kwargs.values()))
        
    def last_updated(self):
        return max(self.cache.keys()) if self.cache else None

    def __call__(self):
        stamp = datetime.datetime.now()
        if self.update():
            value = super(MemCell, self).__call__()
        else:
            value = self.cache[self.last_updated()]
        self.cache[stamp] = value
        return value
    
    def __repr__(self):
        return 'MemCell.last_updated=%s\n'%self.last_updated() + super(MemCell, self).__repr__()



class Const(Cell):
    """calculate once"""
    def update(self):
        return False
    
    def __init__(self, function, args=(), kwargs={}, node=None, config={}):
        if len(args):
            raise TypeError('constant can have no arg parameters')
        if len(kwargs):
            raise TypeError('constant can have no kwarg parameters')
        self.function = function
        self.node = node
        self.config = config
        self.args = ()
        self.kwargs = {}
        
    @classmethod
    def cfg(cls, function, node, **config):
        return cls(function,(),{},node,config)

    @classmethod
    def f(cls, function, *args, **kwargs):
        return cls(function, args, kwargs)

    @classmethod
    def at(cls, function, *args, **kwargs):
        return cls(function, args, kwargs)

    def __call__(self, *_, **__):
        return self.function

    def __repr__(self):
        return 'Const \n%s'%self.function 


class EODCell(MemCell):
    """
    Calculate on update or on a new day. If eod is provided (in seconds), will "reset" at that time
    """

    @property
    def eod(self):
        return timedelta_of_day(self.config.get('eod',0))

    def last_eod(self, date = None):
        """
        returns the previous EOD. If that is greater than last update, we need to re-run function
        """
        now = date or dt.now()
        rtn = dt.today(now) + self.eod
        return rtn if rtn<now else rtn - datetime.timedelta(1)
        
    def update(self):
        last_updated = self.last_updated()
        return last_updated is None or last_updated<self.last_eod() or _update(self.function) or _update(self.args) or _update(list(self.kwargs.values()))

    def __repr__(self):
        return 'EODCell(%s).last_updated=%s \n'%(self.eod, self.last_updated()) + super(EODCell, self).__repr__()

    def metadata(self):
        return super(EODCell, self).metadata() + dict(eod = self.eod)