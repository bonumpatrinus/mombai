import numpy as np
from mombai._containers import as_list, is_array
from mombai._dates import dt
from functools import partial
import datetime


def _per_cell(value, f):
    if is_array(value):
        return [_per_cell(v, f) for v in value]
    elif isinstance(value, Cell):
        return f(value)
    else:
        return value

def _as_asof(asof):
    if asof is None:
        return datetime.datetime.now()
    if isinstance(asof, int):
        return datetime.datetime.now() - datetime.timedelta(seconds = asof)
    if isinstance(asof, datetime.timedelta):
        return datetime.datetime.now() - asof
    return dt(asof)
        

def Hash(value):
    """
    _hash of _hash is the same due to integers not being hashable
    _hash of dict and list are implemented
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, tuple):
        return hash(tuple(value))
    elif isinstance(value, dict):
        return hash(sorted(value.items()))
    else:
        return hash(value)

_call = partial(_per_cell, f = lambda v: v())
        
class Cell(object):
    """
    basic cell, no cache and no persistence
    To make its definition persistent, we will need to register the cell with the big MDS in the sky, Meta Data Store
    :node is actually not just an id, but a whole dict of metadata (potentially)
    :function
    :args
    :kwargs
    are all part of the lazy-function
        
    """
    def __init__(self, node, function, *args, **kwargs):
        self.node = node
        self.function = function
        self.args = args
        self.kwargs = kwargs

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
    
    def update(self):
        """ by default, unless a Cell has been declared to be cached, we assume it is volatile and return True """
        return True

def _update(arg):
    """
    returns the max of all cell.update() in arg
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
    In-Memory Cell
    """
    def __init__(self, node, function, *args, **kwargs):
        super(MemCell, self).__init__(node, function, *args, **kwargs)
        self.cache = self._load_cache()

    def _load_cache(self):
        """ for a file-based or db-based cache, we can implement this"""
        return {}
    
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

