import datetime
from dateutil import parser
from mombai._periods import _ymd2dt, day, Month, BusinessDay
import numpy as np
from functools import reduce
from operator import __add__
from functools import singledispatch
from  mombai._decorators import getargspec
from inspect import FullArgSpec


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(datetime.datetime)
def datetime_to_json(val):
    """Used if *val* is an instance of datetime. to allow json serialisation
    source code: https://hynek.me/articles/serialization/
    >>> import json
    >>> import datetime
    >>> json.dumps({"msg": "hi", "ts": datetime.datetime.now(), 'date' : datetime.date(2019,1,1)}, default=to_serializable)
    """
    return val.isoformat()


"""
This module provides an easy function called dt:
dt is our go-to parsing function for dates
"""

def today(date=None):
    """
    Today, without fraction of day
    """
    now = date or datetime.datetime.now()
    return datetime.datetime(now.year, now.month, now.day)

def _int2dt(arg=0):
    if arg<=1500: ## treat as days from today
        return today() + arg * day        
    if arg<=3000: # treat as year
        return datetime.datetime(arg,1,1)
    elif arg<300000: # treat as excel dates
        return datetime.datetime.fromordinal(arg + 693594)
    elif arg<1095000: #treat as ordinals, less than year 3000
        return datetime.datetime.fromordinal(arg)
    if arg>=10000101 and arg<=30001231: ### treat as yyyymmdd format
        y = arg // 10000
        m = (arg % 10000) // 100
        d = (arg % 100)
        return _ymd2dt(y,m,d)
    else:
        return datetime.datetime.utcfromtimestamp(arg)


def _str2dt(arg):
    """
    The parser has an American tendencies so we check the two formats
    where we can give UK parsing preference.
    >>> import datetime
    >>> D = datetime.datetime
    >>> assert _str2dt('01/09/2001') == D(2001,9,1)
    >>> assert _str2dt('2001-09-01') == D(2001,9,1)
    >>> assert _str2dt('2001 09 01') == D(2001,9,1)
    >>> assert _str2dt('20010901') == D(2001,9,1)
    >>> assert _str2dt('2001 Sep 1st') == D(2001,9,1)
    >>> assert _str2dt('1st Sep 2001') == D(2001,9,1)
    """
    if arg.isdigit():
        return _int2dt(int(arg))
    for txt in [' ', '-', '/']:
        args = arg.split(txt)
        if len(args) == 3 and min([a.isdigit() for a in args]):
            args = [int(a) for a in args]
            if args[0]>50:
                return _ymd2dt(*args)
            elif args[-1]>50:
                return _ymd2dt(*args[::-1])
    return parser.parse(arg)

def dt(*args):
    """
    This is the main date constructor, designed to be able to convert multiple date formats
    >>> import datetime
    >>> D = datetime.datetime
    >>> assert dt('01/09/2001') == D(2001,9,1)
    >>> assert dt('2001-09-01') == D(2001,9,1)
    >>> assert dt('2001 09 01') == D(2001,9,1)
    >>> assert dt('20010901') == D(2001,9,1)
    >>> assert dt('2001 Sep 1st') == D(2001,9,1)
    >>> assert dt('1st Sep 2001') == D(2001,9,1)
    >>> assert dt(20010901) == D(2001,9,1)
    >>> assert dt(2001,9,1) == D(2001,9,1)
    >>> assert dt(D(2001,9,1)) == D(2001,9,1)
    >>> assert dt(D(2001,9,1).toordinal()) == D(2001,9,1)
    >>> assert dt(datetime.date(2001,9,1)) == D(2001,9,1)
    >>> assert dt(0) == today()
    >>> assert dt(-1) == today() - datetime.timedelta(1)
    >>> assert dt(bday) == today() + bday
    >>> assert dt(month) == today() + month
    >>> assert dt(datetime.timedelta(7)) == today() + datetime.timedelta(7)
    >>> assert dt(dt(20000101), month) == dt(2000,2,1)
    >>> assert dt(dt(20000101), 3*year, -month) == dt(2002,12,1)
    
    Known issues: The parser.parse function in dateutil is slightly too lenient, e.g.:
    >>> from dateutil import parser
    >>> assert parser.parse('feb 2012') == datetime.datetime(2012, 2, 10, 0, 0)
    However, nobody is perfect.
    
    """
    if len(args) == 0:
        return datetime.datetime.now()
    if isinstance(args[0], np.datetime64):
        y = args[0].astype('datetime64[Y]').astype(int) + 1970
        month_start = args[0].astype('datetime64[M]')
        m = month_start.astype(int) % 12  + 1
        d = (args[0] - month_start).astype('timedelta64[D]').astype(int) + 1
        hms = datetime.timedelta(args[0].astype(float) % 1)
        return reduce(__add__, args[1:], datetime.datetime(y, m, d) + hms)
    elif isinstance(args[0], datetime.datetime):
        return reduce(__add__, args)
    if len(args) == 3:
        args = [int(a) for a  in args]
        return _ymd2dt(*args)
    elif len(args) == 6:
        args = [int(a) for a  in args]
        return _ymd2dt(args[:3]) + datetime.timedelta(hours=args[3], minutes = args[4], seconds=args[5])
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, datetime.datetime):
            return arg
        elif isinstance(arg, datetime.date):
            return datetime.datetime(arg.year, arg.month, arg.day)
        elif isinstance(arg, str):
            return _str2dt(arg)
        elif isinstance(arg, (BusinessDay, Month, datetime.timedelta)):
            return today() + arg
        else:
            return _int2dt(int(arg)) + datetime.timedelta(float(arg) % 1)


_futs = list('FGHJKMNQUVXZ')
_months = ['january', 'february', 'march','april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
_mmms = [month[:3] for month in _months]
_mms = [('0%i'%i)[-2:] for i in range(1,13)]
_mmm2mm = dict(zip(_mmms, _mms))
_month2mm = dict(zip(_months, _mms))
_fut2mm = dict(zip(_futs, _mms))

def as_mm(month):
    """
    as_mm converts a month into month representation in two-digits month formart.
    >>> assert as_mm(2) == '02'
    >>> assert as_mm('feb') == '02'
    >>> assert as_mm('february') == '02'
    >>> assert as_mm('G') == '02'  ## using Futures code FGHJKMNQUVXZ per month
    >>> assert as_mm(dt(2000,2,2)) == '02'  ## using date.month
    >>> assert as_mm('2nd feb 2002') == '02'  ## using date.month
    """
    if isinstance(month, datetime.datetime):
        return month.strftime('%m')
    elif isinstance(month, int) and month<13 and month>0:
        return ('0%i'%month)[-2:] 
    elif isinstance(month, str):
        if len(month) == 3:
            return _mmm2mm[month.lower()]
        elif len(month) == 1:
            return _fut2mm[month.upper()]
        elif month in _month2mm:
            return _month2mm[month.lower()]
    return as_mm(dt(month))


def timedelta_of_day(*date):
    """
    returns the time from start of day in timedelta format. 
    """
    if len(date) == 1 and isinstance(date[0], datetime.timedelta):
        return date[0] - datetime.timedelta(date[0].days)
    date = dt(*date)
    return date - today(date)

    
def seconds_of_day(*date):
    """
    returns the seconds since start of day for a given date
    >>> from mombai._dates import seconds_of_day
    >>> assert seconds_of_day(dt('2001-04-01T06:00:00')) == 6 * 60 * 60
    >>> assert seconds_of_day('2001-04-01T06:00:00') == 6 * 60 * 60
    """
    day = timedelta_of_day(*date)
    return day.seconds + day.microseconds/1e6

_SECONDS_IN_A_DAY = 24 * 60 * 60
def fraction_of_day(*date):
    """
    returns the intraday time as a fraction of day
    >>> from mombai._dates import fraction_of_day, _SECONDS_IN_A_DAY
    >>> assert fraction_of_day(dt('2001-04-01T06:00:00')) == 0.25
    >>> assert fraction_of_day('2001-04-01T06:00:01') == 0.25 +  1./_SECONDS_IN_A_DAY
    """
    return seconds_of_day(*date)/_SECONDS_IN_A_DAY

def weekday(*date):
    return dt(*date).weekday()

def isoformat(*date):
    return dt(*date).isoformat()


def timestamp(*date):
    return dt(*date).timestamp()

def now(tz = None):
    """
    Weirdly, the datetime.datetime.now getargspec signature is wrong
    """
    return datetime.datetime.now(tz)

dt.now = now 
dt.today = today   
dt.weekday = weekday
dt.isoformat = isoformat
dt.timestamp = timestamp



