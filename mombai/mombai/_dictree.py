"""

Motivation:

dictree contains functionality for another form of dict: a Tree, a dict of dicts. Here is an example such data structure:

>>> tree = dict(teachers = dict(tid01 = dict(name = 'richard', surname = 'feynman', tutor = False, subject = 'physics'),
                                tid02 = dict(name = 'richard', surname = 'dawkins', tutor = True, subject = 'biology'),
                                tid03 = dict(name = 'william', surname = 'shakespear', tutor = True, subject = 'english'),
                                tid04 = dict(name = 'roger', surname = 'penrose', tutor = False, subject = 'maths')
                                ),
    
                students = dict(id01 = dict(name = 'james', surname = 'smith', marks = dict(english = 98, french=85)),
                                id02 = dict(name = 'steve', surname = 'jones', marks = dict(maths = 84, physics=56)),
                                id03 = dict(name = 'richard', surname = 'patel', marks = dict(maths=63, english=45)),
                                id04 = dict(name = 'roger', surname = 'chan', marks = dict(maths=93, english=95))                                    
                                ),
                classes = dict(maths = dict(room = 1, lessons_per_week  = 2),
                               english = dict(room = 2, lessons_per_week  = 3),
                               french = dict(room = 3, lessons_per_week  = 1),
                               physics = dict(room =4, lessons_per_week  = 2)
                               )
                )

We want a declarative method of accessing the data:
    
1) can we select all subjects and their rooms
2) is there an easy way of finding out that french actually has no teacher.
3) can we perform calculations based on teacher, student's marks?

    
"""
from copy import copy
from mombai._dictable import Dict, Dictable
from mombai._dict_utils import dict_concat
    
def _dicts_update(dicts, d):
    """
    >>> dicts = [dict(a=1), dict(b=2)]
    >>> assert _dicts_update(dicts, dict(c=3)) == [dict(a=1, c=3), dict(b=2, c=3)]
    """
    for a in dicts:
        a.update(d)
    return dicts



def tree_to_dicts(tree, match, types = dict):
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
    if isinstance(tree, types): ## it is on the branch       
        if key.startswith('%'):
            return sum([_dicts_update(tree_to_dicts(tree[k], match[1:], types), {key[1:]:k}) for k in tree], [])
        else:
            if key in tree:
                return tree_to_dicts(tree[key], match[1:], types)
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
           

def tree_to_nodes(tree, types = dict):
    """
    >>> tree = dict(teachers = dict(tid01 = dict(name = 'richard', surname = 'feynman', tutor = False),
                                    tid02 = dict(name = 'richard', surname = 'dawkins', tutor = True)
                                    ),
                    students = dict(id01 = dict(name = 'james', surname = 'smith', classes = ['english', 'french']),
                                    id02 = dict(name = 'steve', surname = 'jones', classes = ['maths', 'physics'])))
    
    >>> nodes = tree_to_nodes(tree)
    >>> assert nodes == [('teachers', 'tid01', 'name', 'richard'),
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
        return sum([[(key,) + node for node in tree_to_nodes(tree[key])] for key in tree.keys()], [])
    else:
        return [(tree,)]


def _set_node(tree, node, base, params={}):
    """
    a node is a single tuple ending in the leaf of the tree.
    """
    if len(node)<2:
        raise ValueError('node too short %s'%node)
    res = tree
    for key in node[:-2]:
        if key not in res:
            res[key] = base()
        res = res[key]
    res[node[-2]] = node[-1]
        
def nodes_to_tree(nodes, tree = dict):
    if isinstance(tree, type):
        res = tree()
    else:
        res = copy(tree)
    base = type(res)
    for node in nodes:
        _set_node(res, node, base)
    return res


def _key(key, params={}):
    return params[key[1:]] if key.startswith('%') else key

def _key_to_node(key, params={}):
    return [_key(k, params) for k in (key.split('/') if isinstance(key, str) else key)]


class Dictree(Dict):
    @property
    def nodes(self):
        return tree_to_nodes(self)
    
    def update(self, other):
        nodes = tree_to_nodes(other)
        base = type(self)
        for node in nodes:
            _set_node(self, node, base)
    
    def __getitem__(self, item):
        if '%' in item:
            return Dictable(dict_concat(tree_to_dicts(self, item)))
        res = self
        for key in item.split('/'):
            res = res[key]
        return res
    
    
    