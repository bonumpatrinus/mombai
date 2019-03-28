from mombai._cell import Cell, Hash, MemCell
from mombai._dictable import Dictable
from operator import mul, add
import datetime
import pytest

def test_cell_init():
    c = Cell('hi', add, 'hello', 'world')
    assert c() == 'hello'+'world'
    assert c.update() 
    assert c.id == -7218021353987715941


def test_cell_of_cell():
    c = Cell('hi', add, 'hello', 'world')
    d = Cell('two', mul, c, 2) 
    assert d() == 'helloworldhelloworld'
    assert d.update() 

def test_cell_of_list_of_cells():
    c = Cell('hi', add, 1, 2)
    d = Cell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22

def test_MemCell_update_if_volatile_input():
    c = Cell('hi', add, 1, 2)
    d = MemCell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22
    assert d.update() 
    assert d() == 22
    assert d.update() 

def test_MemCell_not_updated_if_cached_input():
    c = MemCell('hi', add, 1, 2)
    d = MemCell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22
    assert not d.update() 
    assert d() == 22

def test_MemCell_is_cached():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell('hi', lambda value: datetime.datetime.now(), 1)
    assert c.update()
    stamp = c()
    for i in range(100):
        assert not c.update()
        assert c() == stamp
    

def test_MemCell_is_cached_no_args():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell('hi', datetime.datetime.now)
    assert c.update()
    stamp = c()
    for i in range(100):
        assert not c.update()
        assert c() == stamp
