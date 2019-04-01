from mombai._decorators import support_kwargs, try_false, cache
from mombai._dates import dt
from mombai._containers import as_list



class MDS(list):
    """
    this is a simple interface over an index of metadata store presented as dict. Metadata dicts may be filtered using an 'asof' key in the dict

    Simple interface looks like this:    
    >>> self = MDS([dict(a=1,b=2), dict(a=2, c=3), dict(d=1, a=2), dict(a=1, b=3), dict(b=2, c=1), dict(b=1,a=2,c=3)])
    >>> assert dir(self) == ['a', 'b', 'c', 'd']
    >>> assert self['a'] == [1,2]
    >>> assert self.b == MDS([dict(a=1,b=2), dict(a=1, b=3), dict(b=2, c=1), dict(b=1,a=2,c=3)])
    >>> assert self.b.a == MDS([dict(a=1,b=2), dict(a=1, b=3), dict(b=1,a=2,c=3)])
    >>> assert self(b = 1) == MDS([{'b': 1, 'a': 2, 'c': 3}])
    >>> assert self(b = [1,2]) == MDS([{'a': 1, 'b': 2}, {'b': 2, 'c': 1}, {'b': 1, 'a': 2, 'c': 3}])
    >>> assert self(lambda a,b: b<a) == MDS([{'b': 1, 'a': 2, 'c': 3}])
    
    >>> from mombai._dictable import Dictable
    >>> Dictable(self.a.b)
    """
    def __dir__(self):
        return sorted(set(sum([list(d.keys()) for d in self], [])))

    def __getattr__(self, key):
        return MDS([d for d in self if key in d])

    def __getitem__(self, key):
        return sorted(set([d[key] for d in self if key in d]))

    def __call__(self, *args, **kwargs):
        """
        kwargs are filters such as dict(currency = 'USD')
        args are functional filter
        """
        result = self
        for key, value in kwargs.items():
            result = MDS([d for d in result if key in d and d[key] in as_list(value)])
        for arg in args:
            func = try_false(support_kwargs()(arg))
            result = MDS([d for d in result if func(**d)])
        return result            
    
    def __ge__(self, date):
        t = dt(date)
        return MDS([d for d in self if 'asof' not in d or d.get('asof')>=t])

    def __gt__(self, date):
        t = dt(date)
        return MDS([d for d in self if 'asof' not in d or d.get('asof')>t])


    def __le__(self, date):
        t = dt(date)
        return MDS([d for d in self if 'asof' not in d or d.get('asof')<=t])

    def __lt__(self, date):
        t = dt(date)
        return MDS([d for d in self if 'asof' not in d or d.get('asof')<t])
        

