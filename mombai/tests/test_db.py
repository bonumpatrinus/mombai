from mombai._db import db_update, as_db
from tinydb import TinyDB, Query
q = Query()

def test_as_db():
    db = 'db.meta'
    d = as_db(db)
    assert isinstance(d, TinyDB)
    e = as_db(d)
    assert e == d


def test_db_update():
    db = TinyDB('db.test')
    db.purge()
    db.insert(dict(id = 1, name = 'adam', surname = 'smith'))
    db.insert(dict(id = 1, name = 'adam', surname = 'jones'))
    assert len(db.all()) == 2
    
    db.purge()
    db_update(db, dict(id = 1, name = 'adam', surname = 'smith'))
    db_update(db, dict(id = 1, name = 'adam', surname = 'jones'), 'id')
    assert db.all() == [dict(id = 1, name = 'adam', surname = 'jones')]
    db.purge()    

