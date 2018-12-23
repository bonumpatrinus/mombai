from mombai._dates import today, dt
from mombai._periods import day, month, week, bday, Month, BusinessDay, is_eom, is_weekend

import datetime
from dateutil import parser
D = datetime.datetime

def test_month_eom():
    m = Month(1, eom=True)
    date = D(2001,2,28)
    assert is_eom(date)
    assert m + date == D(2001,3,31)    
    assert date - 12 * m == D(2000, 2, 29)    


def test_month_no_roll():
    m = Month(m=1, eom = False)
    date = D(2001,1,30)
    assert date + m == D(2001,2,28)    
    assert date - 12 * m == D(2000, 1, 30)    


def test_month_operations():
    m = Month(1)
    assert 2*m == Month(2)
    assert -m == Month(-1)


def test_month_default():
    assert month == Month(1, eom = False)


def test_bday():
    date = datetime.datetime(2001,3,31) 
    assert is_weekend(date) # actually, Saturday
    assert BusinessDay(convention = 'f').adjust(date) == datetime.datetime(2001,4,2) # Monday following
    assert BusinessDay(convention = 'p').adjust(date) == datetime.datetime(2001,3,30) # Friday previous
    assert BusinessDay(convention = 'm').adjust(date) == datetime.datetime(2001,3,30) # Friday previous as Monday following is April
    assert BusinessDay(convention = 'm').adjust(date-week) == datetime.datetime(2001,3,26) # Monday following

