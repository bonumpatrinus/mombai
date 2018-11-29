from mombai._decorators import getargspec, ARGSPEC, getargs, decorate, cache, try_value, try_back, try_none, try_list, try_str, try_dict, try_nan, try_zero
import numpy as np
from functools import partial
import datetime

def test_getargspec_existing():
    func = lambda x: x
    setattr(func, ARGSPEC, 1)
    assert getargspec(func) == 1
    
def test_getargspec_partial():
    func = lambda x, y=1, *args:1  
    assert getargspec(func) == getargspec(partial(func, y=2))

def test_argspec_general():
    func = lambda x, y=1, *args:1  
    argspec = getargspec(func)
    assert argspec.args == ['x','y']
    assert argspec.varargs == 'args'

def test_argspec_n_zero():
    func = lambda x, y=1, *args:1  
    assert getargs(func) == ['x','y']

def test_argspec_n_one():
    func = lambda x, y=1, *args:1  
    assert getargs(func, n=1) == ['x','y']
    func_no_args = lambda x, y=1:1  
    assert getargs(func_no_args , n=1) == ['y']

def test_decorate():
    def test_function(x,y,z):
        """
        test doc
        """
        pass
    g = decorate(lambda x: x, test_function)
    assert g.__name__ == 'test_function'
    assert g.__doc__.replace('\n','').strip() == 'test doc'
    assert g.argspec.args == ['x','y','z']

def test_try_value():
    f = lambda x: x[0]
    g = try_value('test')(f)
    assert g('hello') == 'h'
    assert g(0) == 'test'
    

def test_try():
    f = lambda x: x[0]
    assert try_none(f)(0) is None
    assert np.isnan(try_nan(f)(0))
    assert try_zero(f)(0) == 0
    assert try_list(f)(0) == []
    assert try_dict(f)(0) == {}
    assert try_str(f)(0) ==''
    assert try_back(f)(0) == 0
    assert try_back(f)(1) == 1

def test_cache_goes_to_cache():
    f = cache(lambda x: datetime.datetime.now())
    assert f.cache == {}
    now = f('now')
    j = 0
    for i in range(100000):
        j = j+i
    then = f('then')
    now_again = f('now')
    assert then>now
    assert now_again==now

def test_cache_function():
    f = cache(lambda x: x)
    assert f.cache == {}
    assert f(4) == 4
    assert f.cache == {((4,), ()): 4}
    

def test_cache_class_method():
    class test(object):
        @cache
        def __call__(self, x):
            return x
    t = test()
    assert t.__call__.cache == {}
    assert t(4) == 4
    assert t.__call__.cache == {((4,), ()): 4}
