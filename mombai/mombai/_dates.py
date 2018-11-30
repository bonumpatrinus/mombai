import datetime
from dateutil import parser
from mombai._periods import _ymd2dt, day, Month, BusinessDay, bday, month, year
from functools import reduce
from operator import __add__
"""
This module provides an easy function called dt:
dt is a parsing function for dates
"""

def today():
    """
    Today, without fraction of day
    """
    now = datetime.datetime.now()
    return datetime.datetime(now.year, now.month, now.day)

def _int2dt(arg=0):
    if arg<=0: ## treat as days from today
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
        raise ValueError("cannot convert %s to a date"%arg)
        

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
    """
    if len(args) == 0:
        return dt(0)
    if isinstance(args[0], datetime.datetime):
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

dt.now = datetime.datetime.now 
dt.today = today   
