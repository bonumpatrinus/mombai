from mombai._dict_utils import dict_zip, dict_concat, dict_merge, dict_append, dict_invert, dict_update_left, dict_update_right, dict_apply

def test_dict_concat():
    dicts = [dict(name = 'James', surname='Smith', salary=100), dict(name = 'Adam', surname='Smith', status='dead'), dict(name = 'Adam', surname='Feingold', salary = 200, status='alive')]    
    result = {'name': ['James', 'Adam', 'Adam'],
              'salary': [100, None, 200],
              'status': [None, 'dead', 'alive'],
              'surname': ['Smith', 'Smith', 'Feingold']}
    assert dict_concat(dicts) == result
    dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    assert dict_concat(dicts) == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1, None, None], 'y': [None, 2, None], 'z': [None, None, 3]}
 
    
def test_dict_zip():
    result = dict({'name': ['James', 'Adam', 'Adam'],
              'salary': [100, None, 200],
              'status': [None, 'dead', 'alive'],
              'surname': ['Smith', 'Smith', 'Feingold']})
    dicts = [dict(name = 'James', surname='Smith', salary=100, status=None), dict(name = 'Adam', surname='Smith', status='dead', salary=None), dict(name = 'Adam', surname='Feingold', salary = 200, status='alive')]    
    assert dict_zip(result) == dicts
    assert dict_concat(dict_zip(result)) == result


def test_dict_apply():
    d = dict(a=1, b=2, c=3)
    assert dict_apply(d, lambda x: x**2) == dict(a=1, b=4, c=9)
    assert dict_apply(d, lambda x: x**2, dict(c=lambda c: c*2)) == dict(a=1, b=4, c=6)
    assert dict_apply(d, lambda x: x**2, dict(c=None)) == dict(a=1, b=4, c=3)

def test_dict_append():
    dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    assert dict_append(dicts) == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1], 'y': [2], 'z': [3]}
    assert dict_append(dicts, keys = 'a') == {'a': [1, 2, 3]}
        
def test_dict_update_right():
    dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    assert dict_update_right(dicts) == {'a': 3, 'b': 6, 'c': 9, 'z' : 3, 'y' : 2,  'x' : 1}
    assert dict_update_right(dicts, keys = 'a') == {'a': 3}

def test_dict_update_left():
    dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    assert dict_update_left(dicts) == {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : 2,  'x' : 1}
    assert dict_update_left(dicts, keys='a') == {'a': 1}    

def test_dict_invert():
    d = dict(a=1,b=1,c=2)
    assert dict_invert(d) == {1 : ['a','b'], 2: ['c']}

def test_dict_merge():
    dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    assert dict_merge(dicts, 'c') ==  {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1, None, None], 'y': [None, 2, None], 'z': [None, None, 3]}
    assert dict_merge(dicts, 'a') == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1], 'y': [2], 'z': [3]}
    assert dict_merge(dicts, 'r') == {'a': 3, 'b': 6, 'c': 9, 'z' : 3, 'y' : 2,  'x' : 1}
    assert dict_merge(dicts, 'l') == {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : 2,  'x' : 1}
    assert dict_merge(dicts, 'l', policies = dict(x = 'a', y='c')) ==  {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : [None, 2, None],  'x' : [1]}

