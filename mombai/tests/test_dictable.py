from mombai._dictable import Dictable, Dict, as_ndarray, vstack, hstack
from mombai._containers import eq
import pytest
import numpy as np

def test_Dictable__init__():
    d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
    assert len(d) == 3
    assert isinstance(d, dict)
    assert np.allclose(d.a, [1,2,3])
    assert np.allclose(d.b, [2,2,2])
    assert isinstance(d.b, np.ndarray)

def test_Dictable__len__():
    import pytest
    d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
    assert len(d) == 3
    with pytest.raises(ValueError):
        d = Dictable(a = [1,2,3], b=2, c=[3,4])

def test_Dictableshape():
    d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
    assert d.shape == (3,3)
    d.d = 1
    assert d.shape == (3,4)

def test_Dictable__iter__():
    d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
    iters = [row for row in d]
    assert iters == [Dict({'a': 1, 'b': 2, 'c': 3}), Dict({'a': 2, 'b': 2, 'c': 4}), Dict({'a': 3, 'b': 2, 'c': 6})]

def test_Dictable_mask():
    d = Dictable(a = [1,2,3], b=2, c=[3,4,6])
    subset = d._mask([True, True, False])
    assert len(subset) == 2
    complement = d._mask([True, True, False], True)
    assert len(complement) == 1        
    resampled = d._mask([2,1,0,0,2])
    assert len(resampled)==5 and np.allclose(resampled.a, [3,2,1,1,3]) 


def test_Dictable__setitem__():
    d = Dictable(a = [1,2,3,4,5])
    assert isinstance(d.a, np.ndarray)
    d['b'] = 1
    assert isinstance(d.b, np.ndarray) and len(d.b) == 5
    with pytest.raises(ValueError):
        d['c'] = [1,2]

def test_Dictable_vectorize():
    d = Dictable(a = [1,2,3,4,5])
    d.b = d[lambda a: range(a)]
    vsq = d._vectorize(lambda x: x**2)
    assert np.allclose(vsq(d.a), d.a**2) 
    vsum = d._vectorize(sum)
    with pytest.raises(TypeError):
        sum(d.b)
    assert np.allclose(vsum(d.b), [0,1,3,6,10]) ## triangular functions        

def test_Dictable_apply():
    d = Dictable(a=1)
    function = lambda x: x+2
    assert np.allclose(d.apply(function, a='x'), [3])
    assert np.allclose(d.apply(function, relabels = dict(a = 'x')), [3])
    d = Dictable(a = range(10))
    assert np.allclose(d.apply(lambda a: a**2), as_ndarray(range(10))**2)

def test_Dictable_exc():
    d = Dictable(a = range(10))
    assert list(d.exc(a = [1,2,3]).a) == [0,4,5,6,7,8,9]
    d = Dictable(a = range(10),  b = [2,1]*5)
    res = d.exc(lambda a,b: b<a, a = range(5,10))
    assert list(res.a) == [0,1,2] and list(res.b) == [2,1,2]

def test_Dictableinc():
    d = Dictable(a = range(10))
    assert len(d.inc(a = [1,2,3])) == 3
    assert list(d.inc(a = [1,2,3]).a) == [1,2,3]
    d = Dictable(a = range(10),  b = [2,1]*5)
    res = d.inc(lambda a,b: b<a, a = range(5))
    assert list(res.a) == [3,4] and list(res.b) == [1,2]

def test_Dictable__getitem__():
    d = Dictable(a = range(10), b=[1,2]*5)
    assert list(d[d.a>5].a) == [6,7,8,9]
    assert list(d[np.array([0,1])].a) == [0,1]
    assert list(d[range(2)].a) == [0,1]
    assert d[0] == Dict(a=0, b=1)
    assert d[-1] == Dict(a=9, b=2)
    res = d[dict(a = [1,2])]
    assert list(res.a) == [1,2]
    res = d[d.a>5, 'a'] 
    assert list(res) == [6,7,8,9]
    res = d[d.a>5, ['a','b']]
    assert isinstance(res, Dictable) and res.shape == (4,2) and list(res.a) == [6,7,8,9]
    res = d[d.a>5, lambda a, b: a+b]
    assert list(res) == [6+1,7+2,8+1,9+2]
    res = d['a']
    assert list(res) == list(range(10))
    res = d['a','b']
    assert isinstance(res, list) and len(res)==2 and list(res[0]) == list(range(10))
    res = d[lambda a,b: a % b]
    assert list(res) == [0,1]*5

def test_Dictable_concat():
    self = Dictable(a=[1,2,3,], b=5, d='hi')
    other = Dictable(a=3, b= ['a','b'], c=1)
    res = self.concat(other)
    assert list(res.d) == ['hi'] * 3 + [None] * 2
    assert list(res.c) == [None] * 3 + [1] * 2

def test_Dictable_concat_multiple():
    self = Dictable(a=[1,2,3,], b=5, d='hi')
    other = Dictable(a=3, b= ['a','b'], c=1)
    another = Dictable(a=4, d= ['a','b'], f=1)
    res = self.concat(other, another)
    assert list(res.d) == ['hi'] * 3 + [None] * 2 + ['a','b']
    assert list(res.c) == [None] * 3 + [1] * 2 + [None, None]
    res = self.concat([other, another])
    assert list(res.d) == ['hi'] * 3 + [None] * 2 + ['a','b']
    assert list(res.c) == [None] * 3 + [1] * 2 + [None, None]

def test_Dictable__add__():
    self = Dictable(a=[1,2,3,], b=5, d='hi')
    other = Dictable(a=3, b= ['a','b'], c=1)
    res = self + other
    assert list(res.d) == ['hi'] * 3 + [None] * 2
    assert list(res.c) == [None] * 3 + [1] * 2

def test_Dictable__sub__():
    self = Dictable(a=[1,2,3,], b=5, d='hi')
    assert (self-'a').keys() == ['b','d']
    assert (self-['a','b']).keys() == ['d']
    
def test_Dictable_sort():
    d = Dictable(a = list('abracadabra'), b=range(11), c = range(0,33,3))
    d.c = d.c % 11
    res = d.sort('c')
    assert list(res.c) == list(range(11))
    d = d.sort('a','c')
    assert list(d.a) == list('aaaaabbcdrr') and list(d.c) == [0,4,8,9,10] + [2,3] + [1] + [7] + [5,6]
    d = d.sort(lambda b: b*3 % 11) ## sorting again by c but using a function
    assert list(d.c) == list(range(11))

def test_Dictable_listby():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    per_a = d.listby('a')
    assert list(per_a.a) == ['a','b','c','d','r'] and list(per_a.b[0]) == [0,3,5,7,10]

def test_Dictable_listby_multiple_keys():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    per_ac = d.listby('a', 'c')
    assert per_ac.keys() == ['a','c','b']
    assert len(per_ac) == 10 and list(per_ac.inc(a = 'a', c='r').b[0]) == [3,10] and list(per_ac.c[:4]) == ['h','p','r','t']
        

def test_Dictable_listby_caclulated_key():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    res = d.listby(dict(bmod2 = lambda b: b % 2), 'a')
    assert res.keys() == ['bmod2','a','b','c']
    assert eq(x = res[0].do(list, 'b','c'), y = Dict(bmod2=0, a = 'a', b = [0,10], c = ['h','r'])) ## top row
    
def test_Dictable_groupby():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    per_a = d.groupby('a')
    assert list(per_a.a) == ['a','b','c','d','r'] and list(per_a.grp[0].b) == [0,3,5,7,10]
        
    per_ac = d.groupby('a', 'c')
    assert per_ac.keys() == ['a','c', 'grp']
    assert len(per_ac) == 10 and list(per_ac.inc(a = 'a', c='r').grp[0].b) == [3,10] and list(per_ac.c[:4]) == ['h','p','r','t']
        
    res = d.groupby(dict(bmod2 = lambda b: b % 2), 'a', grp = 'table')
    assert res.keys() == ['bmod2','a','table']
    assert list(res[0].table.b) == [0,10] ## top row
    
def test_Dictable_unlist():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    da = d.sort('a')
    per_a = d.listby('a')
    d2 = per_a.unlist()
    assert list(da.a) == list(d2.a) and list(da.b) == list(d2.b) and list(da.c) == list(d2.c)

    
def test_Dictable_ungroup():
    d = Dictable(a = list('abracadabra'), b=range(11), c = list('harrypotter'))
    per_a = d.groupby('a')
    da = d.sort('a')
    d2 = per_a.ungroup()
    assert list(da.a) == list(d2.a) and list(da.b) == list(d2.b) and list(da.c) == list(d2.c)
        
def test_Dictable_update():
    x = Dictable(a = [1,2,3])
    y = Dictable(b = [1,2])
    with pytest.raises(ValueError):
        x.update(y)
        

def test_Dictable_table():
    self = Dictable(x = [1,2,3,1,2,3,1,2,3], y = [4,4,4,5,5,5,6,6,6])
    pt = self.pivot_table(index='x',columns='y',values=lambda x,y: x*y, aggfunc=sum)
    assert pt == Dictable({'x': [1, 2, 3], '4': [ 4,  8, 12], '5': [ 5, 10, 15], '6': [ 6, 12, 18]})

def test_Dictable_pair():
    lhs = Dictable(a = [1,2,3,4], b=[1,2,1,2], c=[1,1,2,2])
    rhs = Dictable(a = [4,3,2,1], b=[1,2,1,2], c=[1,1,2,2])
    res = lhs.pair(rhs, ['b','c']) ## join on identical columns!
    assert list(map(list, res.idx)) == [[0, 0 + 4], [1, 1 + 4], [2, 2 + 4], [3, 3+4]] 
    res = lhs.pair(rhs, 'a') ## join reversed columns
    assert list(map(list, res.lhs_idx)) == [[0], [1], [2], [3]] and  list(map(list, res.rhs_idx)) == [[3], [2], [1], [0]] 
    
def test_Dictable_join():
    lhs = Dictable(a = [1,2,3,4], b=[1,2,1,2], c=[1,1,2,2], d='d')
    rhs = Dictable(a = [4,3,2,1], b=[1,2,1,2], c=[1,1,2,2], e='e')
    pair = lhs.pair(rhs, ['b','c']) ## join on identical columns!
    res = lhs._join(pair, rhs, on_left=['b','c'], on_right=['b','c'])
    assert list(map(list, res.a)) == [[1, 4], [2, 3], [3, 2], [4, 1]]
    assert set(res.d) == {'d'} and set(res.e) == {'e'}

def test_Dictable_unpivot():
    d = Dictable(a = range(5))        
    for b in range(5, 10):
        d[str(b)] = d.a * b
    res = d.unpivot('a', 'b', 'axb').do(lambda value: int(value), 'b')
    assert list(res.a * res.b) == list(res.axb)
