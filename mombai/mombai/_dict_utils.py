from mombai._containers import as_type, args_zip, args_to_list, as_list
from functools import reduce
from copy import copy
import numpy as np
import pandas as pd

def dict_zip(d, dict_type=None):
    """
    This function takes a dict whose values are assumed to be in array format, and zip the keys.
    >>> d = dict(a = [1,2,3], b=1, c=['a','b','c'])
    >>> assert dict_zip(d) == [{'a': 1, 'b': 1, 'c': 'a'}, {'a': 2, 'b': 1, 'c': 'b'}, {'a': 3, 'b': 1, 'c': 'c'}]
    """
    dict_type = as_type(d if dict_type is None else dict_type)
    return [dict_type(zip(d.keys(), zipped_value)) for zipped_value in args_zip(*d.values())]


def pass_thru(x):
    return x

def first(x):
    """
    >>> assert first(3) == 3
    >>> assert first(None) == None
    >>> assert first([1,2]) == 1
    """
    res = as_list(x)
    return res[0] if len(res) else None

def last(x):
    """
    >>> assert last(3) == 3
    >>> assert last(None) == None
    >>> assert last([1,2]) == 2
    """
    res = as_list(x)
    return res[-1] if len(res) else None


def dict_apply(d, func=None, funcs=None):
    """
    apply a function to dict values, funcs provide functionality for overriding the function for specific values. 
    
    >>> d = dict(a=1, b=2, c=3)
    >>> assert dict_apply(d, lambda x: x**2) == dict(a=1, b=4, c=9)
    >>> assert dict_apply(d, lambda x: x**2, dict(c=lambda c: c*2)) == dict(a=1, b=4, c=6)
    >>> assert dict_apply(d, lambda x: x**2, dict(c=None)) == dict(a=1, b=4, c=3)
    """
    funcs = funcs or {}
    func = func or pass_thru
    return {key : (funcs.get(key, func) or pass_thru)(value) for key, value in d.items()} 

def dict_concat(*dicts, keys = None):
    """
    >>> dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    >>> assert dict_concat(dicts) == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1, None, None], 'y': [None, 2, None], 'z': [None, None, 3]}
    """ 
    dicts = args_to_list(dicts)
    keys = set(sum([list(d.keys()) for d in dicts], [])) if keys is None else as_list(keys)
    return {key : [d.get(key) for d in dicts] for key in keys}


def dict_append(*dicts, keys = None):
    """
    >>> dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    >>> assert dict_append(dicts) == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1], 'y': [2], 'z': [3]}
    >>> 
    """
    dicts = args_to_list(dicts)
    keys = set(sum([list(d.keys()) for d in dicts], [])) if keys is None else as_list(keys)
    return {key : [d[key] for d in dicts if key in d] for key in keys}
        
def dict_update_right(*dicts, keys = None, **kwargs):
    """
    >>> dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    >>> assert dict_update_right(dicts) == {'a': 3, 'b': 6, 'c': 9, 'z' : 3, 'y' : 2,  'x' : 1}
    """
    dicts = args_to_list(dicts)[::-1]
    keys = set(sum([list(d.keys()) for d in dicts], [])) if keys is None else as_list(keys)
    res = {}
    for key in keys:
        for d in dicts:
            if key in d:
                res[key] = d[key]
                break
    return res

def dict_update_left(*dicts, keys = None, **kwargs):
    """
    >>> dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    >>> assert dict_update_left(dicts) == {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : 2,  'x' : 1}
    >>> 
    """
    dicts = args_to_list(dicts)
    keys = set(sum([list(d.keys()) for d in dicts], [])) if keys is None else as_list(keys)
    res = {}
    for key in keys:
        for d in dicts:
            if key in d:
                res[key] = d[key]
                break
    return res


def dict_invert(d):
    """
    >>> d = dict(a=1,b=1,c=2)
    >>> assert dict_invert(d) == {1 : ['a','b'], 2: ['c']}
    """
    res = {}
    for key, value in d.items():
        res.setdefault(value, []).append(key)
    return res


_merge_policies= {'l' : dict_update_left, 'r' : dict_update_right, 'c' : dict_concat, 'a' : dict_append, 'left' : dict_update_left, 'right' : dict_update_right, 'concat' : dict_concat, 'append' : dict_append}


def _dict_update(left, right):
    left.update(right)
    return left


def dict_merge(dicts, policy='c', dict_type = None, policies=None, **kwargs):
    """
    When we have two (or more) dicts where we want to merge them. If the keys don't overlap, there is no problems.
    However, if there are two identical keys, we need to have a policy:
    'left': pick the left most dict with the key
    'right': pick the right most dict with key (equivaluen to dict.update)
    'append': per each key, create a list of the values whichever dict have this key
    'concat': per each key, create a fixed length list, each dict will provide d.get(key)
    
    policies allow us to apply very specific policy per specific keys
    >>> dicts = [{'a': 1, 'b': 4, 'c': 7, 'x' : 1}, {'a': 2, 'b': 5, 'c': 8, 'y' : 2}, {'a': 3, 'b': 6, 'c': 9, 'z' : 3}]
    >>> assert dict_merge(dicts, 'c') ==  {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1, None, None], 'y': [None, 2, None], 'z': [None, None, 3]}
    >>> assert dict_merge(dicts, 'a') == {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9], 'x': [1], 'y': [2], 'z': [3]}
    >>> assert dict_merge(dicts, 'r') == {'a': 3, 'b': 6, 'c': 9, 'z' : 3, 'y' : 2,  'x' : 1}
    >>> assert dict_merge(dicts, 'l') == {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : 2,  'x' : 1}
    >>> assert dict_merge(dicts, policy = 'l', x = 'a', y='c') ==  {'a': 1, 'b': 4, 'c': 7, 'z' : 3, 'y' : [None, 2, None],  'x' : [1]}
    >>>
    """
    dicts = args_to_list(dicts)
    dict_type = as_type(dict_type or (type(dicts[0]) if len(dicts)>0 else dict))
    keys = set(sum([list(d.keys()) for d in dicts], []))
    policies = policies or {}
    policy_to_keys = dict_invert({key: policies.get(key,policy) for key in keys})
    for p, keys in policy_to_keys.items():
        _merge_policies[p[0].lower()](dicts, keys = keys)
    policy_to_dict = {p : _merge_policies[p[0].lower()](dicts, keys = keys, **kwargs) for p, keys in policy_to_keys.items()}
    return reduce(_dict_update, policy_to_dict.values(), dict_type())


def _dicts_update(dicts, d):
    """
    >>> dicts = [dict(a=1), dict(b=2)]
    >>> assert _dicts_update(dicts, dict(c=3)) == [dict(a=1, c=3), dict(b=2, c=3)]
    """
    for a in dicts:
        a.update(d)
    return dicts



def tree_to_dicts(tree, match):
    """
    We are going to define a tree as an opject that is of the form: dicts-of-dicts e.g.
    >>> tree = dict(teachers = dict(tid01 = dict(name = 'richard', surname = 'feynman', tutor = False),
                                    tid02 = dict(name = 'richard', surname = 'dawkins', tutor = True)
                                    ),
                    students = dict(id01 = dict(name = 'james', surname = 'smith', classes = ['english', 'french']),
                                    id02 = dict(name = 'steve', surname = 'jones', classes = ['maths', 'physics'])))
    We would like to get a "flat" structure as  list of dicts.
    >>> match = 'students/%student_id/name/%student_name'
    as we traverse down the tree. 
    >>> assert tree_to_dicts(tree, match) ==[{'student_name': 'james', 'student_id': 'id01'}, {'student_name': 'steve', 'student_id': 'id02'}]
    >>> assert tree_to_dicts(tree, ['teachers','%tid', 'tutor', '%is_tutor']) == [{'is_tutor': False, 'tid': 'tid01'}, {'is_tutor': True, 'tid': 'tid02'}]
    >>> assert tree_to_dicts(tree, 'teachers/tid01/name/richard') == [{}] ## exists 
    >>> assert tree_to_dicts(tree, 'teachers/tid01/name/adam') == [] ## not exists
    >>> assert tree_to_dicts(tree, 'teachers/tid01/name/richard/whatever') == []
    """
    match = match.split('/') if isinstance(match, str) else match
    if len(match) == 0:
        return [{}]
    key = match[0]
    if isinstance(tree, dict): ## it is on the branch       
        t = dict(tree)
        if key.startswith('%'):
            return sum([_dicts_update(tree_to_dicts(t[k], match[1:]), {key[1:]:k}) for k in tree], [])
        else:
            if key in tree:
                return tree_to_dicts(t[key], match[1:])
            else:
                return []
    else: # tree is the leaf
        if len(match) > 1: 
            return []
        elif key.startswith('%'):
            return [{key[1:] : tree}]
        elif key == tree:
            return [{}]
        else:
            return []
           

def tree_items(tree, types = dict):
    """
    Given a tree like structure, we enumerate all the paths (nodes). This is 
    
    >>> tree = Dictree(teachers = dict(tid01 = dict(name = 'richard', surname = 'feynman', tutor = False),
                                    tid02 = dict(name = 'richard', surname = 'dawkins', tutor = True)
                                    ),
                    students = dict(id01 = dict(name = 'james', surname = 'smith', classes = ['english', 'french']),
                                    id02 = dict(name = 'steve', surname = 'jones', classes = ['maths', 'physics'])))
    
    >>> items = tree_items(tree)
    >>> assert items  == [('teachers', 'tid01', 'name', 'richard'),
                                     ('teachers', 'tid01', 'surname', 'feynman'),
                                     ('teachers', 'tid01', 'tutor', False),
                                     ('teachers', 'tid02', 'name', 'richard'),
                                     ('teachers', 'tid02', 'surname', 'dawkins'),
                                     ('teachers', 'tid02', 'tutor', True),
                                     ('students', 'id01', 'name', 'james'),
                                     ('students', 'id01', 'surname', 'smith'),
                                     ('students', 'id01', 'classes', ['english', 'french']),
                                     ('students', 'id02', 'name', 'steve'),
                                     ('students', 'id02', 'surname', 'jones'),
                                     ('students', 'id02', 'classes', ['maths', 'physics'])]
    
    """
    if isinstance(tree, types):
        return sum([[(key,) + node for node in tree_items(tree[key])] for key in tree.keys()], [])
    else:
        return [(tree,)]

def _is_pattern(pattern):
    if not isinstance(pattern, str):
        return False
    pattern = pattern.split('/')
    if not max([p.startswith('%') for p in pattern]):
        return False
    if max(['%' in p[1:] for p in pattern]):
        return False
    return True

def _as_pattern(pattern):
    return pattern.split('/') if isinstance(pattern, str) else pattern    


def _key_to_item(key, params={}):
    return params[key[1:]] if key.startswith('%') else key

def _pattern_to_item(pattern, params={}):
    """
    A pattern is of the form 'students/%name/%surname/classes/%subject/%grade'
    or is of the form ['students', '%name', '%surname', 'classes', '%subject', '%grade']
    We provide a dict to evaluate parameters such as %name
    
    >>> assert _pattern_to_item('students/%name/%surname/classes/%subject/%grade', dict(name = 'adam', surname = 'smith', subject = 'economics', grade=100)) == ['students', 'adam', 'smith', 'classes', 'economics', 100]
    >>> assert _pattern_to_item(['students', '%name', '%surname'], dict(name = 'adam', surname = 'smith', subject = 'economics', grade=100)) == ['students', 'adam', 'smith']
    
    """
    return [_key_to_item(k, params) for k in _as_pattern(pattern)]


def _set_item(tree, item, base=dict):
    """
    in-place setting an item in a tree, Ab item is a single tuple ending in the leaf of the tree.
    """
    if len(item)<2:
        raise ValueError('item too short %s'%item)
    res = tree
    for key in item[:-2]:
        if key not in res:
            res[key] = base()
        res = res[key]
    res[item[-2]] = item[-1]
        
def items_to_tree(items, tree = dict, raise_if_duplicate = True):
    """
    >>> d = dict(a = 1, b=2)
    >>> items = d.items()
    >>> assert items_to_tree(items) == d
    
    >>> d = dict(teachers = dict(tid01 = dict(name = 'richard', surname = 'feynman', tutor = False),
                                    tid02 = dict(name = 'richard', surname = 'dawkins', tutor = True)
                                    ),
                    students = dict(id01 = dict(name = 'james', surname = 'smith', classes = ['english', 'french']),
                                    id02 = dict(name = 'steve', surname = 'jones', classes = ['maths', 'physics'])))
    >>> items = tree_items(d)
    >>> t = items_to_tree(items)
    >>> assert t == d 
    >>> assert t['teachers']['tid02']['tutor'] == True
    >>> t = items_to_tree([('teachers', 'tid02', 'tutor', False)], t)
    >>> assert t['teachers']['tid02']['tutor'] == False    
    """
    if raise_if_duplicate and len(set([tuple(node[:-1]) for node in items]))<len(items):
        raise ValueError('items are not unique and overwriting each other')
    
    if isinstance(tree, type):
        res = tree()
    else:
        res = copy(tree)
    base = type(res)
    for item in items:
        _set_item(res, item, base)
    return res


def data_and_columns_to_dict(data=None, columns=None):
    """
    data is assumed to be a list of records (i.e. horizontal) rather than column inputs
    >>> from mombai._compare import eq
    >>> data = [[1,2],[3,4],[5,6]]; columns = ['a','b']
    >>> assert data_and_columns_to_dict(data, columns) == {'a': (1, 3, 5), 'b': (2, 4, 6)}
    
    We can convert from pandas:
    >>> df = pd.DataFrame(data=data, columns = columns)
    >>> dfa = df.set_index('a')
    >>> assert data_and_columns_to_dict(df)['a'] == [1,3,5]
    >>> assert data_and_columns_to_dict(dfa)['a'] == [1,3,5]
    >>>
    >>> data = [['a','b'],[1,2],[3,4],[5,6]]; columns = None
    >>> assert list(data_and_columns_to_dict(data, columns)['a']) == [1,3,5]
    >>>
    >>> data = dict(a = [1,3,5], b = [2,4,6]); columns = None
    >>> assert data_and_columns_to_dict(data) == data
    >>> assert data_and_columns_to_dict(data.items()) == data
    >>> assert data_and_columns_to_dict(zip(data.keys(), data.values())) == data
    """
    if data is None and columns is None:
        return {}
    elif columns is not None:
        if isinstance(columns, str):
            if isinstance(data, dict) and _is_pattern(columns):
                return dict_concat(tree_to_dicts(data, columns))
            else:
                return {columns : as_list(data)}
        else:
            
            return dict(zip(columns, args_zip(*data)))
    else:
        if isinstance(data, str):
            data = pd.read_csv(data)
        if isinstance(data, pd.DataFrame):
            if data.index.name is not None:
                data = data.reset_index()
            return data.to_dict('list')
        if isinstance(data, list):
            if len(data) == 0:
                return {}
            elif min([isinstance(record, tuple) and len(record) == 2 for record in data]): ## tuples
                return dict(data)
            elif min([isinstance(d, dict) for d in data]):
                return dict_concat(data)
            else:
                return data_and_columns_to_dict(data[1:], data[0])
        else:
            return dict(data)
