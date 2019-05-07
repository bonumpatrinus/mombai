from mombai._graph import XCL, DAG, jp, Hash, _is_ref, _from_id
from mombai._cell import Cell, Const, MemCell
from operator import add
from copy import copy, deepcopy
from mombai._dictable import Dictable
import networkx as nx


def test_XCL_at():
    g = XCL()
    g['a'] = Cell(Dictable(a = [1,2,3]))
    g['a']()
    g['b'] = Cell.f('@a', b = lambda a: a+1)
    assert g.at['b']() == Dictable(a = [1,2,3], b =[2,3,4])
    
def test_Cell_at():
    g = XCL()
    g['a'] = 1
    b = lambda a: a+1
    assert Cell.at(b) == Cell.f(b, a = '@a')
    g['b'] = Cell.at(b)
    assert g['b'] == Cell.cfg(Cell.f(b, a = '@a'), 'b')
    g['c'] = Cell.at(lambda a, b: a + b)
    assert g.at['b']() == 2
    assert g.at['c']() == 3
#    g.at['c']
#    g.from_id()['c'].kwargs
#    g.at#q = Query()
#db.insert(dict(a = 1, b = 'key'))
#db.insert(dict(a = 2, b = 'key'))
#db.all()
#db.remove
#db.remove(q.a.exists())
#db.search(q.a >= 1)
#db.all()

def test_DAG_set_get_del_nodes():
    g = DAG()
    g['a'] = 1
    assert list(g.nodes) == ['a']
    g['b'] = 2
    assert list(g.nodes) == ['a', 'b']
    assert g._node['a'] == dict(value = 1)
    assert g._node['b'] == dict(value = 2)
    assert g['a'] == 1
    assert g['b'] == 2
    del g['a']
    assert list(g.nodes) == ['b']


def test_DAG__contains__():
    g = DAG()
    g['a'] = 1
    assert 'a' in g
    assert 'b' not in g
    g['b'] = 2 
    g['a','b'] = 'ab'
    assert 'b' in g
    assert ('a','b') in g    
    assert ('a','c') not in g    


def test_DAG_set_get_del_edges():
    g = DAG()
    g['a'] = 1
    g['b'] = 2
    g['c'] = 3
    g['a','b'] = 'ab'
    assert list(g.edges) == [('a', 'b')]
    g['a','c'] = 'ac'
    assert list(g.edges) == [('a', 'b'), ('a', 'c')]    
    assert g.edges['a', 'b'] == dict(value = 'ab')
    assert g['a', 'b'] == 'ab'
    assert g['a', 'c'] == 'ac'
    del g['a', 'b']
    assert list(g.edges) == [('a', 'c')]    


def test_DAG_remove_node():
    g = DAG()
    g['a'] = 1
    g['b'] = 2
    g['c'] = 3
    g['d'] = 4
    g.remove_node(['a','b'])
    assert list(g.nodes) == ['c','d']
    g.remove_node()
    assert list(g.nodes) == []


def test_DAG_remove_edge():
    g = DAG()
    g['a'] = 1
    g['b'] = 2
    g['c'] = 3
    g['d'] = 4
    g['e'] = 5
    for u in 'abcde':
        for v in 'abcde':
            if v>u : 
                g[u,v]=u+v            
    assert list(g.edges) == [('a', 'b'), ('a', 'c'),('a', 'd'),('a', 'e'), ('b', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'd'), ('c', 'e'), ('d', 'e')]
    g.remove_edge([('a', 'b'), ('a', 'c'),('a', 'd'),('a', 'e')])    
    assert list(g.edges) == [('b', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'd'), ('c', 'e'), ('d', 'e')]
    g.remove_edge()
    assert list(g.edges) == []
    

def test_XCL_key():
    g = XCL()
    g['a'] = Cell(1)
    assert list(g.nodes) == ['a']
    g[0] = Cell(0) ## index by an integer 0
    ha = 'a'
    assert list(g.nodes) == ['a', '0']
    assert g._key('a') == ha
    assert g._key('@a') == ha
    assert g._key(ha) == ha
    assert g._key(g['a']) == ha
    assert g._key(('a', 0)) == (ha, '0')
    assert '@a' in g
    assert 'a' in g

def test_XCL_key_dict():
    g = XCL()
    cell = Cell(0, node = dict(a=1, b=2, c=3))
    g[cell] = cell
    assert g[ {'a': 1, 'b': 2, 'c': 3}] == cell
    del g[ {'a': 1, 'b': 2, 'c': 3}]
    assert len(g.nodes) == 0

def test_XCL_add():
    from operator import add
    a = Cell.cfg(0, 0)
    b = Cell.cfg(1, 1)
    c = Cell.cfg(2, 2)
    d = Cell(add, (a, b), {}, 3)
    e = Cell(add, (c, d), {}, 4)
    f = Cell.f(sum, [a,b,c,d,e], 0)    
    assert a() == 0
    assert b() == 1
    assert c() == 2
    assert d() == 1
    assert e() == 3
    assert f() == 7
    g = XCL()
    g += a
    assert a in g
    g +=[b,c]
    assert b in g and c in g
    g += f
    assert d in g and e in g and f in g
    f.inputs
    assert g[f]() == 7

def test_XCL_to_from_id():
    a = Cell.cfg(0, 0)
    b = Cell.cfg(1, 1)
    c = Cell.cfg(2, 2)
    d = Cell(add, (a, b), node = 3)
    e = Cell(add, (c, d), node = 4)
    f = Cell(sum, ([a,b,c,d,e], 0), node = 5)   
    g = XCL()
    g+= f
    h = g.to_id()
    i = h.from_id()
    assert g[f].args[0] == [a,b,c,d,e]
    assert h[f].args[0] == ['@0', '@1', '@2', '@3', '@4']
    assert i[f].args[0] == [i[a],i[b],i[c],i[d], i[e]]

def test_XCL_to_from_id_different_id_types():
    a = Cell.cfg(0, 0)
    b = Cell.cfg(1, 1)
    c = Cell.cfg(2, 2)
    d = Cell(add, (a, b), node = 3)
    e = Cell(add, (c, d), node = 4)
    f = Cell(sum, ([a,b,c,d,e], 0), node = 5)   
    g = XCL()
    g+= f
    h = g.to_id()
    i = h.from_id()
    assert g[f].args[0] == [a,b,c,d,e]
    assert h[f].args[0] == ['@0','@1','@%s'%c.id, '@3' ,'@4']
    assert i[f].args[0] == [i[a],i[b],i[c],i[d], i[e]]


def test_XCL_to_json():
    a = Cell.cfg(0, 0)
    b = Cell.cfg(1, 1)
    c = Cell.cfg(2, 2)
    d = Cell(add, (a, b), node = 3)
    e = Cell(add, (c, d), node = 4)
    f = Cell(sum, ([a,b,c,d,e], 0), node = 5)   
    g = XCL()
    g+= f
    j = g.to_json()
    assert '["@0", "@1", "@2", "@3", "@4"]' in j 
    assert '["@2", "@3"]' in j
    assert '["@0", "@1"]' in j
    assert 'add' in j
    assert 'sum' in j
    i = XCL.from_json(j)
    assert i[f]() == 7
    assert i[f].args[0] == [i[a],i[b],i[c],i[d], i[e]]

def test_XCL_to_table():
    a = Cell.cfg(0, 0)
    b = Cell.cfg(1, 1)
    c = Cell.cfg(2, 2)
    d = Cell(add, [a, b], node = 3)
    e = Cell(add, [c, d], node = 4)
    f = Cell(sum, ([a,b,c,d,e]), node = 5)    
    g = XCL()
    g+= f
    j = g.to_table()
    assert len(j) == len(g._node) and j.keys() == ['node_id', 'node', 'function', 'args', 'kwargs']    
    
def test_XCL_replace():
    g = XCL()
    g['a'] = 1
    g['b'] = Cell.at(lambda a: a + 1)
    g['c'] = Cell.at(lambda a, b: a+b)
    g['d'] = Cell.at(lambda a,b,c: a+b+c)
    g['e'] = Cell.at(lambda a,b,c,d: a+b+c+d)    
    h = g.from_id()
    assert h['c']() == 3
    assert h['d']() == 6    
    new = dict(c = Cell.at(lambda a, b: a + 2*b))
    i = g.replace(new)
    assert i.at['c']() == 5
    assert i.at['d']() == 8
    i = g.replace(c = Cell.cfg(lambda a, b: a * b, 'cc'))
    assert sorted(i.nodes) == ['a','b','cc','d','e']
    assert i['cc']() == 2
    assert i['d']() == 5
    assert i['d'].kwargs['c'] == i['cc']
