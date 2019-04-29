import numpy as np
import jsonpickle as jp
import datetime
from mombai._containers import as_list, is_array
from mombai._decorators import Hash
from mombai._dates import dt, fraction_of_day, seconds_of_day
from mombai._dict import Dict
from functools import partial


def _per_cell(value, f):
    if is_array(value):
        return type(value)([_per_cell(v, f) for v in value])
    elif isinstance(value, Cell):
        return f(value)
    else:
        return value

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

class Cell(object):
    """
    basic cell, no cache and no persistence
    To make its definition persistent, we will need to register the cell with the big MDS in the sky, Meta Data Store

    :node is actually not just an id, but a whole dict of metadata (potentially)
    :function
    :args
    :kwargs
    are all part of the lazy-function

    :params are not exposed for the basic Cell
    deepcopy(Cell('a', 1))
        
    """
    def __init__(self, node, function, *args, **kwargs):
        self.node = node
        if not callable(function) and len(args) == 0 and len(kwargs) == 0: ## a hack to allow simple cells Cell('constant', 1)
            self.args = (function,)
            self.function = passthru
            self.kwargs = {}
        else:
            self.function = function
            self.args = args
            self.kwargs = kwargs
        self.params = Dict()

    @property
    def id(self):
        """
        A hash of self.node
        """
        return Hash(self.node)

    def __call__(self, *_, **__):
        """
        first of all evaluates the parents and then evaluates the self.function itself
        """
        function = _call(self.function)
        args = (_call(arg) for arg in self.args)
        kwargs = {key: _call(value) for key, value in self.kwargs.items()}
        return function(*args, **kwargs)
    
    def calc(self):
        """ by default, unless a Cell has been declared to be cached, we assume it is volatile and return True """
        return True
    
    def __eq__(self, other):
        """
        >>> a = Cell('a', 1)
        >>> b = Cell('a', 1)
        >>> assert a == b
        """
        return type(other) == type(self) and self.node == other.node and self.function == other.function and self.args == other.args and self.kwargs == other.kwargs

    def __repr__(self):
        if not callable(self.function):
            return '@%s: %s'%(self.id, self.function)

        else:
            return '@%s: %s'%(self.id, getattr(self.function, '__name__', self.function)) + '(' + (('%s'%list(self.args))[1:-1] if self.args else '') + (', **%s'%self.kwargs if self.kwargs else '') + ')'
    
    def metadata(self):
        res = Dict(self.node) if isinstance(self.node, dict) else Dict(node = self.node)
        res['json']  = jp.encode(self)
        res['type'] = jp.encode(type(self))
        res['function'] = jp.encode(self.function)
        return res
        
def _update(arg):
    """
    returns the max of all cell.calc() in arg
    """
    if isinstance(arg, Cell):
        return arg.calc()
    elif is_array(arg):
        for a in arg:
            if _update(a):
                return True
    return False        

class MemCell(Cell):
    """
    In-Memory caching Cell
    """
    def __init__(self, node, function, *args, **kwargs):
        super(MemCell, self).__init__(node, function, *args, **kwargs)
        self.cache = self._load_cache()

    def _load_cache(self):
        """ for a file-based or db-based cache, we can implement this"""
        return Dict()

    def _to_cache(self, cache):
        """ for a file-based or db-based cache, we can implement this"""
        return cache
    
    def calc(self):
        return len(self.cache) == 0 or _update(self.function) or _update(self.args) or _update(list(self.kwargs.values()))
        
    def last_updated(self):
        return max(self.cache.keys()) if self.cache else None

    def __call__(self):
        stamp = datetime.datetime.now()
        if self.calc():
            value = super(MemCell, self).__call__()
        else:
            value = self.cache[self.last_updated()]
        self.cache[stamp] = value
        return value
    
    def __repr__(self):
        return 'MemCell.last_updated=%s | '%self.last_updated() + super(MemCell, self).__repr__()



class EODCell(MemCell):
    """
    Calculate on update or on a new day. If eod is provided (in seconds), will "reset" at that time
    """
    def __init__(self, node, function, *args, **kwargs):
        eod = kwargs.pop('eod', 0)
        super(EODCell, self).__init__(node, function, *args, **kwargs)
        self.params.eod = eod

    def last_eod(self, date = None):
        """
        returns the previous EOD. If that is greater than last update, we need to re-run function
        """
        now = date or dt.now()
        rtn = dt.today(now) + datetime.timedelta(seconds = self.params.eod)
        return rtn if rtn<now else rtn - datetime.timedelta(1)
        
    def calc(self):
        last_updated = self.last_updated()
        return last_updated is None or last_updated<self.last_eod() or _update(self.function) or _update(self.args) or _update(list(self.kwargs.values()))

    def __repr__(self):
        return 'EODCell(eod=%s).last_updated=%s | '%(self.params.eod, self.last_updated()) + super(MemCell, self).__repr__()

    def metadata(self):
        return super(EODCell, self).metadata() + dict(eod = self.params.eod)