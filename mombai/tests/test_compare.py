from mombai._compare import eq, Cmp, cmp, Index
from numpy import nan, array, int64
import numpy as np
import pandas as pd


def test_eq():
    assert eq(np.nan, np.nan)
    assert eq(np.array([np.array([1,2]),2]), np.array([np.array([1,2]),2]))
    assert eq(np.array([np.nan,2]),np.array([np.nan,2]))    
    assert eq(dict(a = np.array([np.array([1,2]),2])) ,  dict(a = np.array([np.array([1,2]),2])))
    assert eq(dict(a = np.array([np.array([1,np.nan]),np.nan])) ,  dict(a = np.array([np.array([1,np.nan]),np.nan])))
    assert eq(np.array([np.array([1,2]),dict(a = np.array([np.array([1,2]),2]))]), np.array([np.array([1,2]),dict(a = np.array([np.array([1,2]),2]))]))
    
    class FunnyDict(dict):
        pass
    assert not eq(dict(a = 1), FunnyDict(a=1))    
    assert 1 == 1.0
    assert eq(1, 1.0)
    assert eq(np.inf, np.inf)
    assert not eq(np.inf, -np.inf)
    assert not eq(np.inf, np.nan)
    assert eq(pd.DataFrame([1,2]), pd.DataFrame([1,2]))
    assert eq(pd.DataFrame([np.nan,2]), pd.DataFrame([np.nan,2]))
    assert eq(pd.DataFrame([1,np.nan], columns = ['a']), pd.DataFrame([1,np.nan], columns = ['a']))
    assert not eq(pd.DataFrame([1,np.nan], columns = ['a']), pd.DataFrame([1,np.nan], columns = ['b']))
    assert not eq(pd.DataFrame([1,np.nan], columns = ['a'], index=['a','b']), pd.DataFrame([1,np.nan], columns = ['a'], index=[0,1]))

def test_Cmp():
    assert Cmp(None)<Cmp(1)
    assert Cmp(1.0) == Cmp(1)
    assert sorted([1,2,3,None, 'a'], key = Cmp) == [None, 1, 2, 3, 'a']
    assert sorted([1,2,3,None, 'a', 2.0, 1.0], key = Cmp) ==  [None, 1, 1.0, 2, 2.0, 3, 'a']
    assert sorted([1,2.0, 1, 1.0, np.nan, 0, 0.0, 1], key = Cmp) == sorted([1,2.0, 1, 1.0, np.nan, 0, 0.0, 1])
    assert sorted([1,2.0, None, 1, 'a', 1.0, np.nan, 0, 0.0, 1], key = Cmp) ==  [None, 0, 0.0, 1, 1, 1.0, 1, 2.0, np.nan, 'a']

def test_cmp():
    assert cmp(nan, 1) == 1 
    assert cmp([1,nan],[1,2]) == 1 
    assert cmp([1,2,3], y = [1,2,3]) == 0
    assert cmp(np.array([1,2,3]), np.array([1,2,3])) == 0
    assert cmp(x = np.array([1,2,nan]), y = np.array([1,2,3])) == 1
    x = np.array([1,2,[1,2,nan]], dtype = 'object')
    y = np.array([1,2,[1,2,3]], dtype = 'object')
    assert cmp(x,y) ==1 
    assert cmp(0,None) == 1 
    assert cmp(0,1) == -1 
    assert cmp(2,1) == 1 
    assert cmp(2,2) == 0 
    assert cmp(2,2.) == 0 
    assert cmp(np.nan,2.) == 1     


def test_Index_single_mixed_values():
    values = [3,6,2,1.0, 3.0, 4,None, nan, ]
    i = Index([values])
    assert eq(i._sorted[0], np.array([None, 1.0, 2, 3, 3., 4, 6, nan]))
    assert eq(i.argsort, array([6, 3, 2, 0, 4, 5, 1, 7]))
    assert eq(i.unique, [array([None, 1.0, 2, 3, 4, 6, nan])])
    assert i.group(values) ==  [[None], [1.0], [2], [3, 3.0], [4], [6], [nan]]


def test_Index_single_mixed_values_with_strings():
    values = [3,6,2,1.0, 'b', 3.0, 4,None, nan, 'a']
    i = Index([values])
    assert eq(i._sorted[0], array([None, 1.0, 2, 3, 3.0, 4, 6, nan, 'a', 'b'], dtype=object))
    assert eq(i.argsort, array([7, 3, 2, 0, 5, 6, 1, 8, 9, 4], dtype=int64))
    assert eq(i.unique, [array([None, 1.0, 2, 3, 4, 6, nan, 'a', 'b'])])
    assert i.group(values) ==  [[None], [1.0], [2], [3, 3.0], [4], [6], [nan], ['a'], ['b']]
