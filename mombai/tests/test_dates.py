from mombai._dates import today, dt
from mombai._periods import day, month, week, bday, BusinessDay, Month, is_eom, year, quarter

import datetime
from dateutil import parser
D = datetime.datetime

def test_today():
    res = datetime.datetime.today()
    if res.hour<23:
        assert today() == D(res.year, res.month, res.day)


def test_units():
    assert day == datetime.timedelta(1)
    assert week == datetime.timedelta(7)
    assert year == Month(12)
    assert quarter == Month(3)


def test_dt():
    assert dt('01/09/2001') == D(2001,9,1)
    assert dt('2001-09-01') == D(2001,9,1)
    assert dt('2001 09 01') == D(2001,9,1)
    assert dt('20010901') == D(2001,9,1)
    assert dt('2001 Sep 1st') == D(2001,9,1)
    assert dt('1st Sep 2001') == D(2001,9,1)
    assert dt(20010901) == D(2001,9,1)
    assert dt(2001,9,1) == D(2001,9,1)
    assert dt(D(2001,9,1)) == D(2001,9,1)
    assert dt(D(2001,9,1).toordinal()) == D(2001,9,1)
    assert dt(datetime.date(2001,9,1)) == D(2001,9,1)
    assert dt(0) == today()
    assert dt(-1) == today() - day
    assert dt(bday) == today() + bday
    assert dt(month) == today() + month
    assert dt(datetime.timedelta(7)) == today() + datetime.timedelta(7)

def test_dt_today():
    assert dt.today() == today()

