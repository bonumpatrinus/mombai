from mombai._dict import Dict, slist, Dictattr
import pytest

d = Dict(a=1, b=2, c=3, d=4)    

def test_Dictattr():
    a = Dictattr(a = 1, b = 2)
    assert a['a'] == a.a
    a.c = 3
    assert a['c'] == 3
    del a.c
    assert list(a.keys()) == ['a','b']
    assert a['a','b'] == [1,2]
    assert a[['a','b']] == Dictattr(a = 1, b=2)
    assert not a == dict(a=1,b=2)


def test_Dict__eq__empty():
    assert Dict() == Dict()
    
def test_Dict__add__():
    e = Dict(d = 5, e = 10)
    assert d + e == Dict(a =1, b=2,c=3, d=5, e=10)
    assert e + d == Dict(a =1, b=2,c=3, d=4, e=10)

def test_dict__sub__():
    assert d - 'a' == Dict(b=2,c=3,d=4)
    assert d - ['a','b'] == Dict(c=3,d=4)
    assert d - ['a','x'] == d-'a'

def test_Dict__and__():
    assert d & 'a' == Dict(a=1)
    assert d & ['a','b'] == Dict(a=1,b=2)
    assert d & ['a','x'] == d & 'a'

def test_Dict_keys():
    assert isinstance(d.keys(), slist)
    assert d.keys() == slist(['a','b','c','d'])
    
def test_Dict_values():
    assert isinstance(d.values(), list)

def test_dict_do():
    d = Dict(a=1, b=2, c=3, d=4)
    assert d.do(lambda value: value*2, 'a','b','c') == Dict(a = 2, b=4, c=6, d = 4)
    assert d.do(lambda value: value*2, ['a','b','c']) == Dict(a = 2, b=4, c=6, d = 4)
    assert d.do(lambda value: value*2, 'd') == d + Dict(d = 8)    
    e = d.do(str, 'a', 'b', 'c') -'d'
    assert e == Dict(a = '1', b='2', c='3')
    e = e.do(lambda value, a: value+a, ['b', 'c'])
    assert e == Dict(a = '1', b='21', c='31')        
    assert e.do([int, lambda value, a: value-int(a)], 'b','c') == Dict(a = '1', b=20, c=30)


def test_dict_relabel():
    assert d.relabel(b = 'bb', c=lambda value: value.upper()) == Dict(a = 1, bb=2, C=3, d=4)
    assert d.relabel(relabels = dict(b='bb', c=lambda value: value.upper())) == Dict(a=1, bb=2, C=3, d=4)


def test_dict__call__():
    d = Dict(a=1, b=2, c=3, d=4)    
    assert d(e = 0) == d+dict(e=0)
    assert d(e = lambda d, b, c: d-b) == d + dict(e=2)
    assert d(e = lambda d, b, x=1: d-b+1) == d + dict(e=3)


def test_Dict_dir():
    assert 'a' in dir(d)
    assert 'b' in dir(d)
    assert 'c' in dir(d)
    assert 'd' in dir(d)


def test_Dict__getattr__():
    assert d.a == d['a']
    with pytest.raises(AttributeError):
        d.x


def test_Dict__setattr___():
    d.e = 1
    assert 'e' in d.keys()
    assert d['e'] == 1
    del d['e']
    assert d == Dict(a=1, b=2, c=3, d=4)    


def test_Dict__delattr___():
    d.e = 1
    assert 'e' in d.keys()
    del d.e
    assert d == Dict(a=1, b=2, c=3, d=4)    
    
def test_Dict_raises_AttributeError():
    with pytest.raises(AttributeError):
        del d.not_there

def test_hasattr_works():
    assert not hasattr(d, 'not_there')
    assert hasattr(d, 'a')


def test_Dict_apply():
    d = Dict(a=1, b=2, c=3, d=4)    
    assert d.apply(lambda a,b: a+b) == 3
    assert d.apply(lambda a,b,x=1: a+b+x) == 4
    with pytest.raises(TypeError):
        d.apply(lambda x: x)
        
def test_Dict_relabel():
    d = Dict(a=1, b=2, c=3)
    assert d.relabel(relabels = lambda x: x*2) == Dict(aa=1, bb=2, cc=3)
    assert d.relabel(b='BB') == Dict(a=1, BB=2, c=3)
    assert d.relabel(b='BB', c = lambda x: x+'rap') == Dict(a=1, BB=2, crap=3)
    

def test_Dict_mask():
    d = Dict(a = None, b=None, c='1')
    x = d.mask(None, a=0, b=1, c=2)
    assert x == Dict(a=0, b=1, c='1')
    y = d.mask(lambda value: value is None, a=0, b=lambda value, c: c*2, c=2)
    assert y == Dict(a=0, b='11', c='1')

def test_Dict_where():
    d = Dict(a = None, b=None, c='1')
    x = d.where('not me', a=0, b=1, c=2)
    assert x == Dict(a=0, b=1, c=2)
    y = d.mask(lambda value: value is not None, a=0, b=1, c=2)
    assert y == Dict(a=None, b=None, c=2)

def test_Dict_where_advance():
    d = Dict(a=None, b=None, c='1')
    assert d.where(None, float) == Dict(a=None, b=None, c=1.0)
    assert d.where(lambda value: value is not None, 0) == Dict(a=0, b=0, c='1')
    assert d.where(lambda value: value is not None, 0, b='exception') == Dict(a=0, b='exception', c='1')
    
def test_Dict_mask_advance():
    d = Dict(a=None, b=None, c='1')
    assert d.mask(None, 0) == Dict(a=0, b=0, c='1')
    assert d.mask(None, 0).mask(lambda value: isinstance(value, str), float) == Dict(a=0, b=0, c=1.0)
    assert d.mask(lambda value: value is None, 0) == Dict(a=0, b=0, c='1')
    assert d.mask(None, 0, b='exception') == Dict(a=0, b='exception', c='1')
    e = Dict(a = 1, b=10, c=100, d = -999)
    assert e.mask(-999, 0).mask(lambda v: isinstance(v, int), str) == Dict(a = '1', b='10', c='100', d = '0')
