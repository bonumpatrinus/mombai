from tinydb import TinyDB, Query
from mombai._containers import as_list

def as_db(db):
    return TinyDB(db) if isinstance(db, str) else db

q = Query()
    
def db_update(db, data, key = 'id'):
    """
    TinyDb allows multiple values inserted 
    >>> from tinydb import TinyDB, Query
    >>> db = TinyDB('db.test')
    >>> db.purge()
    >>> db.insert(dict(id = 1, name = 'adam', surname = 'smith'))
    >>> db.insert(dict(id = 1, name = 'adam', surname = 'smith'))
    >>> assert len(db.all()) == 2
    
    If we want to have a "key-value" store, then use db_update. There is a package https://github.com/schapman1974/tinymongo to replicate Mongo. 
    Here we create the simplest functionality
    
    >>> db = TinyDB('db.test')
    >>> db.purge()
    >>> db.insert(dict(id = 1, name = 'adam', surname = 'smith'))
    >>> db_update(db, dict(id = 1, name = 'adam', surname = 'jones'), 'id')
    >>> assert db.all() == [dict(id = 1, name = 'adam', surname = 'jones')]
    >>> db.purge()    
    """
    db = as_db(db)
    key = as_list(key)
    old = db.search(*[getattr(q,k)==data[k] for k in key])
    if len(old)>1:
        raise ValueError('more than one item of data per this key %s' %({k: data[k] for k in key}))
    if len(old)==1:
        if old == data:
            return True
        db.remove(*[getattr(q,k)==data[k] for k in key])
    try:
        db.insert(data)
        return True
    except Exception:
        for o in old:
            db.insert(o)
    return False


if False:
    import h5py
    import numpy as np
    from mombai import *
    import pandas as pd
    f = h5py.File('D:/myfile.hdf5','w')
    df = pd.DataFrame([dt.now()]*10)
    f['test'] = pd.DataFrame([dt.now()]*10)
    df.to_hdf('D:/myfile.hdf5', 'test', mode = 'w')
    pd.read_hdf('D:/myfile.hdf5', 'test')
    
    
    f.create_dataset("test", (100,), data = ['a'] * 100, dtype="S1")
    f['test'] = list(['a' * i for i in range(100)])
    f.close()
    
    
    g = h5py.File('D:/myfile.hdf5','r')
    np.array(g['test'])
    g.close()
