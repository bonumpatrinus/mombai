from operator import mul, add
import datetime
from mombai._dict import Dict
from mombai._cell import Cell, MemCell, HDFCell, Const, getargspec, passthru, _call
from mombai._dates import dt
from tinydb import TinyDB, Query
import pytest
import jsonpickle as jp

q = Query()


def test_Cell_init_with_const():
    x = Cell(1)
    assert x() == 1

def test_Cell_init_with_wrong_args():
    with pytest.raises(TypeError):
        Cell(lambda a: a, 1)

def test_Cell_init_with_wrong_kwargs():
    with pytest.raises(TypeError):
        Cell(lambda a, b: (1,), 1)


def test_Cell_init_with_f():
    x = Cell.f(lambda a, b: a+b, 'a','b')
    assert x() == 'ab'

def test_Cell_init_with_f_fails_with_no_kwargs():
    with pytest.raises(TypeError):
        Cell.f(lambda a:21)

def test_Cell_init_with_f_fails_with_too_many_kwargs():
    with pytest.raises(TypeError):
        Cell.f(lambda a:21, a = 1, b=2)

def test_Cell_init_with_f_failts_with_duplicate_kwargs():
    with pytest.raises(TypeError):
        Cell.f(lambda a, b: a+b, 1, a=2)
    
def test_Cell_init_with_at():
    c = Cell.at(lambda a, b: a+b)
    assert c() == '@a@b'


def test_Cell_init_():
    a = Cell(lambda a, b: a +b, kwargs = dict(a = 1, b=2), node = 'ab', config = dict(test = 1))
    assert a() == 3
    assert a.id == 'ab'
    assert a.config == Dict(test = 1)
    assert a.node == 'ab'


def test_Cell_init_with_Cell_with_params():
    x = Cell(lambda a, b: a +b, kwargs = dict(a = 1, b=2), node = 'ab', config = dict(test = 1))
    y = Cell.f(x, a = 2,b=3)
    assert y() == 5
    assert y.config == Dict() and y.node is None

def test_Cell_init_with_Cell_with_no_params():
    x = Cell(lambda a, b: a +b, kwargs = dict(a = 1, b=2), node = 'ab', config = dict(test = 1))
    y = Cell(x)
    assert y == x
    assert y() == 3

def test_Cell_cell_of_cell():
    x = Cell.at(lambda a,b:a+b)
    y = Cell.at(lambda a,b:a+b, a = x)
    z = Cell.at(lambda a,b:a+b, a = x, b = y)
    assert x() == '@a@b'
    assert y() == '@a@b@b'
    assert z() == '@a@b@a@b@b'

def test_cell_of_list_of_cells():
    c = Cell.f(add, 1, 2)
    d = Cell.f(sum, [c,c,c,c], 10)
    assert d() == 22

def test_MemCell_update_if_volatile_input():
    c = Cell.f(add, 1, 2)
    d = MemCell.f(sum, [c,c,c,c], 10)
    assert d() == 22
    assert d.update() 
    assert c.update()
    assert d() == 22
    assert d.update() 

def test_MemCell_not_updated_if_cached_input():
    c = MemCell.f(add, 1, 2)
    d = MemCell.f(sum, [c,c,c,c], 10)
    assert d() == 22
    assert not d.update() 
    assert d() == 22

def test_MemCell_is_cached():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell.f(lambda value: datetime.datetime.now(), 1)
    assert c.update()
    stamp = c()
    for i in range(100):
        assert not c.update()
        assert c() == stamp
    

def test_MemCell_is_cached_no_args():
    """
    we use a deliberately volatile function but declare it as cached
    """
    c = MemCell.f(dt.now)
    assert c.update()
    stamp = c()
    for i in range(100):
        assert not c.update()
        assert c() == stamp


def test_cell__add__():
    c = Cell(1, node = dict(a=1, b=2))
    assert (c + dict(b=3)).node == Dict(a = 1, b= 3)
    assert (c + dict(c=3)).node == Dict(a = 1, b= 2, c=3)    

def test_cell__sub__():
    c = Cell(1, node = dict(a=1, b=2))
    assert (c - 'a').node == Dict(b=2)
    assert (c - ['a']).node == Dict(b=2)


def test_cell_relabel():
    c = Cell(None, node = dict(a=1, b=2))
    assert c.relabel(a = 'ayala', b='boaz').node == Dict(ayala=1, boaz=2)


def test_cell_inputs():
    f = lambda a,b,c: a+b+c
    c = Cell.f(f, 1, b=2, c=3)
    assert c.inputs == [f, 1, 2, 3]


def test_cell_cfg():
    f = lambda a, b: a+b
    c = Cell.cfg(f, 'name', eod = 1, ccy = dict(usd = '10y', gbp = '5y'))
    assert c() == '@a@b'
    assert c.config == Dict(eod = 1, ccy = dict(usd = '10y', gbp = '5y'))
    assert c.node == 'name'

    
def test_cell_reconfig():
    f = lambda a, b: a+b
    c = Cell.cfg(f, 'name', eod = 1, ccy = dict(usd = '10y', gbp = '5y'))
    d = c.reconfig(eod = 2, ccy = c.config.ccy, new = 1)
    assert d.config == Dict(eod = 2, ccy = dict(usd = '10y', gbp = '5y'), new = 1)

def test_cell_config_update():
    f = lambda a, b: a+b
    c = Cell.cfg(f, 'name', eod = 1, ccy = dict(usd = '10y', gbp = '5y'))
    d = c.config_update(ccy = dict(usd = '7y', mxn = '1y'), new = 1)
    assert d.config == Dict(eod = 1, ccy = dict(usd = '7y', gbp = '5y', mxn = '1y'), new = 1)


def test_Cell_to_id():
    a = Cell.cfg(1, 'a', eod = 1)
    b = Cell.cfg(2, 'b', eod = 2)
    c = Cell.f(add, a, b).reconfig(eod = 3)    
    assert c()
    d = c.to_id()
    assert d() == '@a@b'
    assert d.config == Dict(eod = 3)
    assert d.node is None


def test_Cell_to_db():
    cell = Cell.cfg(lambda a: a+1, 'test', ccy='USD')
    db = TinyDB('db.test')    
    db.purge()
    cell.to_db('db.test')
    cell.to_db(db)
    assert len(db.all())==1
    db.purge()
    def f(a,b):
        return a+b
    cell = Cell.cfg(f, dict(ccy = 'USD', tenor = '10Y'))
    with pytest.raises(ValueError):
        cell.to_db()
    cell.to_db('db.test')
    r = db.search(q.node.ccy == 'USD' and q.id == cell.id)[0]
    d = jp.decode(r['json'])
    assert d() == '@a@b'
    db.purge()
    c = Cell.at(f)
    with pytest.raises(ValueError):
        c.to_db()
