import numpy as np
from mombai._containers import as_list, is_array
from mombai._decorators import Hash
from mombai._dates import dt, fraction_of_day, seconds_of_day
from mombai._dict import Dict
from functools import partial
import datetime


def _per_cell(value, f):
    if is_array(value):
        return [_per_cell(v, f) for v in value]
    elif isinstance(value, Cell):
        return f(value)
    else:
        return value

_call = partial(_per_cell, f = lambda v: v())


def _as_asof(asof):
    if asof is None:
        return datetime.datetime.now()
    if isinstance(asof, int):
        return datetime.datetime.now() - datetime.timedelta(seconds = asof)
    if isinstance(asof, datetime.timedelta):
        return datetime.datetime.now() - asof
    return dt(asof)
        
       
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
        if not callable(function):
            return function
        args = (_call(arg) for arg in self.args)
        kwargs = {key: _call(value) for key, value in self.kwargs.items()}
        return function(*args, **kwargs)
    
    def calc(self):
        """ by default, unless a Cell has been declared to be cached, we assume it is volatile and return True """
        return True

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
    In-Memory Cell
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

