from mombai._dictable import Dictable
import datetime
from dateutil import relativedelta
from dateutil.relativedelta import *
from dateutil.rrule import rrule, WEEKLY, SA, SU, MO, TU, WE,TH, FR

T0 = datetime.datetime(1970,1,1)
T1 = datetime.datetime(2300,1,1)


_history = Dictable(date = list(rrule(WEEKLY, wkst=SU, byweekday=(MO,TU,WE,TH, FR), dtstart = T0, until = T1)))
def as_holiday(dates):
    hols = _history.xor(dict(date = dates))    
    return hols
      
day = datetime.timedelta(1)
week = 7 * day

def _as_ym(y,m):
    """
    converts a y,m into a proper y,m with m in [1,12]
    assert _as_ym(2001,-1) == (2000,11)
    assert _as_ym(2001,0) == (2000,12)
    assert _as_ym(2001,13) == (2002,1)
    """
    y+= (m-1)//12
    m = ((m-1) % 12) + 1
    return (y,m)        

def _ymd2dt(y,m,d):
    """
    The function handles months and days which are not within the 1-12 and 1-31 range
    There allows the user to specify e.g. (2001,1,0) as the last day in the month previous to Jan 2001,so:
    A 0 month is interpreted as 12 of previous year and 0th day is interpreted as the last day of previous month.
    so:
    >>> assert dt(2000,0,1) == datetime.datetime(1999,12,1)
    >>> assert dt(2000,0,0) == datetime.datetime(1999,11,30)
    >>> assert dt(2000,12,0) == datetime.datetime(2000,11,30)
    >>> assert dt(2000,12,-1) == datetime.datetime(2000,11,29)
            
    >>> assert _ymd2dt(2001,0,1) == datetime.datetime(2000,12,1)
    >>> assert _ymd2dt(2001,0,0) == datetime.datetime(2000,11,30)
    >>> assert _ymd2dt(2003,1,1) == datetime.datetime(2003,1,1)
    >>> assert _ymd2dt(2003,0,1) == datetime.datetime(2002,12,1)
    """
    y,m = _as_ym(y,m)
    return datetime.datetime(y,m,1) + (d-1)*day

def is_eom(date):
    return (date+day).month != date.month

def is_weekend(date, locale=None):
    if locale is None: 
        return date.weekday()>=5
    elif locale.lower() == 'israel':
        return date.weekday() in [4,5]
    else:
        raise ValueError('This locale not implemented')

class Month(object):
    """
    Month is a class allowing us to add months to dates. Month arithmetic is a tricky question:
    What is 2001-01-30 + 1 month?
    what is 2001-01-31 + 1 month?
    
    For US Treasuries, there is an eom convention: if your current date is EOM, then adding a month must take you to EOM too.
    This is the "eom" flag. Otherwise, we revert to dateutil.relativedelta
    
    >>> assert datetime.datetime(2001,1,30) + Month(1) == datetime.datetime(2001,2,28)
    >>> assert datetime.datetime(2001,1,31) + Month(1, eom = True) == datetime.datetime(2001,2,28)
    """
    def __init__(self, m=1, eom = False):
        self.m = m
        self.eom = eom
    def copy(self):
        return Month(m = self.m, eom = self.eom)
    
    def __add__(self, date):
        eom = _ymd2dt(date.year, date.month+1+self.m, 0)
        if self.eom and is_eom(date):
            return eom
        res = _ymd2dt(date.year, date.month + self.m, date.day)
        if res <= eom:
            return res
        else:
            return eom

    __radd__ = __add__     
    
    def __mul__(self, months):
        res = self.copy()
        res.m = self.m * months
        return res

    __rmul__ = __mul__
    
    def __neg__(self):
        return self * (-1)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __repr__(self):
        return "Months(%i)"%self.m + (' using eom' if self.eom else '')

    def __eq__(self, other):
        return type(other) == type(self) and other.m == self.m and other.eom == self.eom





class BusinessDay(object):
    """
    In determining schedule of e.g. a swap, we first start at the maturity and work backwards at 3M or 6M intervals.    
    We then need to adjust the coupon dates for non-business days.
    There are three conventions: 
    1) following: move to the next business day
    2) previous: move to the previous business day
    3) modified following (default): move to following unless it is a new month, in which case, move to previous

    >>> date = datetime.datetime(2001,3,31) 
    >>> assert is_weekend(date) # actually, Saturday
    >>> assert BusinessDay(convention = 'f').adjust(date) == datetime.datetime(2001,4,2) # Monday following
    >>> assert BusinessDay(convention = 'p').adjust(date) == datetime.datetime(2001,3,30) # Friday previous
    >>> assert BusinessDay(convention = 'm').adjust(date) == datetime.datetime(2001,3,30) # Friday previous as Monday following is April
    >>> assert BusinessDay(convention = 'm').adjust(date-week) == datetime.datetime(2001,3,26) # Monday following
    
    Implementing a full Holiday Calendar is possible but we would then think about optimization
    """
    def __init__(self, b=0, convention = 'm', hols = None):
        self.b = b
        self.convention = self._as_convention(convention)
        self.hols = hols

    def _as_convention(self, convention):
        if convention is None:
            convention = 'm'
        convention = convention[0].lower()
        if convention not in ('p', 'f', 'm'):
            raise ValueError('only prev/following/modified following are allowed')
        return convention

    def is_holiday(self, date):
        return is_weekend(date)
            
    def copy(self):
        return BusinessDay(b = self.b,  convention = self.convention)

    def __mul__(self, days):
        res = self.copy()
        res.b = self.b * days
        return res

    __rmul__ = __mul__
    
    def __add__(self, date):
        """
        for the time being, we ignore the holidays calenday
        """
        date = self.adjust(date)
        weeks = self.b // 5
        remainder = self.b % 5
        res = date + weeks * week
        while remainder>0:
            res += day
            remainder -=1
            while self.is_holiday(res):
                res += day
        while remainder<0:
            res -= day
            remainder +=1
            while self.is_holiday(res):
                res -= day
        return res
            
    __radd__ = __add__ 

    def __neg__(self):
        return self * (-1)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def _following(self, date):
        return date + (7-date.weekday()) * day
    
    def _previous(self, date):
        return date - (date.weekday()-4) * day

    
    def adjust(self, date):
        if self.is_holiday(date):
            if self.convention.startswith('p') :
                return self._previous(date)
            elif self.convention.startswith('f'):
                return self._following(date)
            elif self.convention.startswith('m'):
                following = self._following(date)
                if following.month!= date.month:
                    return self._previous(date)
                else:
                    return following
            else:
                raise ValueError('adjust convention %s not implemented'%self.convention)
        else:
            return date
    def __repr__(self):
        return "BusinessDays(%i) %s"%(self.b, self.convention)

bday = BusinessDay(1)
month = Month(1)
year = Month(12)
quarter = Month(3)