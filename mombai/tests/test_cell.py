from operator import mul, add
import datetime
from mombai._dict import Dict
from mombai._cell import Cell, MemCell

def test_cell_init():
    c = Cell('hi', add, 'hello', 'world')
    assert c() == 'hello'+'world'
    assert c.calc() 
    assert c.id == hash(c.node)


def test_cell_of_cell():
    c = Cell('hi', add, 'hello', 'world')
    d = Cell('two', mul, c, 2) 
    assert d() == 'helloworldhelloworld'
    assert d.calc() 

def test_cell_of_list_of_cells():
    c = Cell('hi', add, 1, 2)
    d = Cell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22

def test_MemCell_update_if_volatile_input():
    c = Cell('hi', add, 1, 2)
    d = MemCell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22
    assert d.calc() 
    assert d() == 22
    assert d.calc() 

def test_MemCell_not_updated_if_cached_input():
    c = MemCell('hi', add, 1, 2)
    d = MemCell('sum', sum, [c,c,c,c], 10) 
    assert d() == 22
    assert not d.calc() 
    assert d() == 22

def test_MemCell_is_cached():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell('hi', lambda value: datetime.datetime.now(), 1)
    assert c.calc()
    stamp = c()
    for i in range(100):
        assert not c.calc()
        assert c() == stamp
    

def test_MemCell_is_cached_no_args():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell('hi', datetime.datetime.now)
    assert c.calc()
    stamp = c()
    for i in range(100):
        assert not c.calc()
        assert c() == stamp


