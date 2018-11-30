from mombai._containers import as_type, args_zip, args_to_list, as_list
from functools import reduce

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
