from mombai._decorators import getargspec, ARGSPEC, getargs, decorate, cache, try_value, try_back, try_none, try_list, try_str, try_dict, try_nan, try_zero, support_kwargs, relabel
from mombai._decorators import list_loop, dict_loop, callattr, callitem, Hash
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

def test_hash_seed_zero():
    if hash('a') != -7583489610679606711:
        raise ValueError('hash function is using a random seed, please set PYTHONHASHSEED=0 in environment variables')
    

def test_decorate():
    def test_function(x,y,z):
        """
        test doc
        """
        pass
    g = decorate(lambda x: x, test_function)
    assert g.__name__ == 'test_function'
    assert g.__doc__.replace('\n','').strip() == 'test doc'
    assert getattr(g, ARGSPEC).args == ['x','y','z']

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
    assert t(4) == 4
    assert t.cache == {('__call__', (4,), ()): 4}

def test_relabel():
    relabels = dict(b='BB', c = lambda value: value.upper())
    assert relabel('a', relabels) == 'a'
    assert relabel('b', relabels) == 'BB'
    assert relabel('c', relabels) == 'C'
    assert relabel('do nothing',  None) == 'do nothing'

    assert relabel(dict(a = 1, b=2, c=3), relabels) == {'a': 1, 'BB': 2, 'C': 3}
    assert relabel(['a', 'b', 'c'], relabels) == ['a', 'BB', 'C']
    assert relabel(('a', 'b', 'c'), relabels) == ('a', 'BB', 'C')
    assert relabel(None, relabels) is None
    
    assert relabel(dict(a = 1, b=2, c=3), lambda label: label*2) == {'aa': 1, 'bb': 2, 'cc': 3}
    assert relabel('label_in_text_is_renamed', 'market') == 'market_in_text_is_renamed'

    func = lambda a, b, c : a+b+c
    assert getargs(func) == ['a', 'b', 'c']
    relabeled_func = relabel(func, relabels)
    assert getargspec(relabeled_func).args == ['a', 'BB', 'C']
    assert relabeled_func(1,2,3) == 6
    assert relabeled_func(a=1,BB=2,C=3)==6
    
def test_support_kwargs():
    relabels = dict(b='BB', c = lambda value: value.upper())
    func = lambda a, b, c : a+b+c
    assert getargs(func) == ['a', 'b', 'c']
    relabeled_func = support_kwargs(relabels)(func)
    assert getargspec(relabeled_func).args == ['a', 'BB', 'C']
    assert relabeled_func(1,2,3) == 6
    assert relabeled_func(a=1,BB=2,C=3)==6
    assert relabeled_func(a=1,BB=2,C=3, some_other_stuff =3)==6


def test_support_kwargs_with_varkw():
    function = lambda x, **kwargs : ''.join(['x'*x] + [key*value for key, value in kwargs.items()]) 
    support_kwargs()(function)(x=1, b=2, c=3) == 'xbbccc'
    support_kwargs(dict(x='a'))(function)(a=1, b=2, c=3) == 'xbbccc'
    support_kwargs(dict(x='a', y='b'))(function)(x=1, b=2, c=3) == 'xbbccc' # we never relabel kwargs. so 'b' is passed to the function rather than 'y'        

def test_Hash():
    d = dict(a = 1, b=2)
    assert Hash(d) == hash(tuple(sorted(d.items())))
    lst = [1,2,3]
    assert Hash(lst) == hash(tuple(lst))
    i = 5
    assert Hash(i) == i

def test_callitem():
    d = dict(a = lambda x,y,z=1: x+y+z, b = lambda x, y, z=2: x*y*z)
    assert callitem(d, 'a', 3, 2) == 6 #args
    assert callitem(d, 'a', x=3, y=2) == 6 #kwargs
    assert callitem(d, 'a', 3, y=2) == 6 #mixture
    assert callitem(d, 'b', 3, 2) == 12


def test_callattr():
    from mombai._dict import Dict
    a = lambda x,y,z=1: x+y+z; b = lambda x, y, z=2: x*y*z
    d = Dict(a=a, b=b, c=1, d=2)
    e = Dict(a=a, b=b, c=1, d=2, e = 3)
    assert callattr(d, 'sub', 'a') == Dict(b = b, c=1, d=2)
    assert callattr(d, 'call', e = lambda c, d: c+d) == e
    assert callattr(d, 'a', 3, 2) == 6
    assert callattr(d, 'a', x=3, y=2) == 6
    assert callattr(d, 'a', 3, y=2) == 6

