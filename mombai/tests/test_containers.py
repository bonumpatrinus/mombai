from mombai._containers import ordered_set, slist, args_to_list, args_to_dict, args_zip, is_array, as_list, as_ndarray, as_array, eq, Cmp, cmp
import numpy as np
import pandas as pd
import pytest


def test_ordered_set():
    assert ordered_set([1,2,3]) - 1 == ordered_set([2,3])
    assert ordered_set([1,2,3]) + 4 == ordered_set([1,2,3,4])
    assert ordered_set([1,2,3]) + [3,4] == ordered_set([1,2,3,4])
    assert ordered_set([1,2,3]) & [3,4] == ordered_set([3])
    assert ordered_set([1,2,3]) | [3,4] == ordered_set([1,2,3,4])
    assert ordered_set([1,2,3]) % [3,4] == ordered_set([1,2])

    
def test_slist():
    assert slist([1,2,3]) - 1 == [2,3]
    assert slist([1,2,3]) + 4 == [1,2,3,4]
    assert slist([1,2,3]) + [3,4] == [1,2,3,4]
    assert slist([1,2,3]) & [3,4] == [3]


def test_as_list():
    assert as_list(None) == []
    assert as_list(5) == [5]
    assert as_list('str') == ['str']
    assert as_list((1,2)) == [1,2]
    assert as_list(np.array([1,2])) == [1,2]
    assert as_list(dict(a=1)) == [dict(a=1)]
    assert as_list(dict(a=1).keys()) == ['a']
    assert as_list(range(3)) == [0,1,2]


def test_as_array():
    assert as_array(None) == []
    assert as_array(5) == [5]
    assert as_array('str') == ['str']
    assert as_array((1,2)) == (1,2)
    assert np.allclose(as_array(np.array([1,2])) , np.array([1,2]))
    assert as_array(dict(a=1)) == [dict(a=1)]
    assert as_array(range(3)) == range(3)


def test_as_ndarray():
    assert np.all(np.array([]) == as_ndarray(None))
    assert np.all(np.array([5]) == as_ndarray(5)) 
    assert np.all(np.array(['str']) == as_ndarray('str'))
    assert np.all(np.array([1,2]) == as_ndarray((1,2)))
    assert np.all(np.array([1,2]) == as_ndarray(np.array([1,2])))
    assert np.all(np.array(['a']) == as_ndarray(dict(a=1).keys()))
    assert np.all(np.array([0,1,2])== as_ndarray(range(3)))
    assert eq(x = as_ndarray(['a',[1,2]]), y = np.array(['a',[1,2]], dtype='object'))


def test_args_to_dict():
    assert args_to_dict(('a','b',dict(c='d'))) == dict(a='a',b='b',c='d')
    assert args_to_dict([dict(c='d')]) == dict(c='d')
    assert args_to_dict(dict(c='d')) == dict(c='d')
    assert args_to_dict(['a','b',dict(c='d'), dict(e='f')]) == dict(a='a',b='b',c='d', e='f')
    with pytest.raises(ValueError):
        args_to_dict(['a','b',lambda c: c]) == dict(a='a',b='b',c='d', e='f')


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
