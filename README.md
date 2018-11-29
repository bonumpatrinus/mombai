# mombai
Mombai is a collection of utilities to allow fast research &amp; development
The basic structures are:

## date utilities
1) classes ```BusinessDay``` and ```Month``` that allow direct additions to datetime.datetime and implement standard financial date convention, so that expressions like ```date + 3 * month``` actually work.
2) function ```dt``` that is an all-round "accept anything and cast to datetime.datetime"

## dict utilities
```dict_invert, dict_merge, dict_zip``` and ```dict_apply``` are basic dict manipulations that allow us to: reverse key/values, merge two or more dicts together, handling keys overlap. ```dict_zip``` is equivalent to zip() but for a dict and ```dict_apply``` is equivalent to ```pd.DataFrame.apply```

## Dict
Dict inherits from dict and introduces an important concept: we can access function of keys as well as the keys. Themselves. This, simple extension actually has far reaching implications, in allowing us from thinking of Dict as a static container of values, to a full 'calculation graph'. Instead of:

```
def func(a, b):
  c = a+b
  d = c+b
  e = c+d
  return e 

func(5,2)
```
we can move to interactive coding:
```
x = Dict(a = 5, b = 2)
x.c = x[lambda a, b: a+b]
x.d = x[lambda c, b: c+b]
x.e = x[lambda c, d: c+d]
```
or indeed:
```
x = x(c = lambda a, b: a+b, d = lambda c, b: c+b, e = lambda c, d: c+d)
```

## Dictable (Dict and a Table)
Dictable inherits from Dict so has this nice calculation framework but is also a column-based table, with each key being a vector of equal length. The code still looks like:
```
x = x(c = lambda a, b: a+b)(d = lambda c, b: c+b)(e = lambda c, d: c+d)
```
but now this is applied per-row allowing us to track multiple experiments at the same time.
Since Dictable is a table, we also support fast ```apply, sort, groupby, pivot_table, merge``` and filtering as per pandas. It is important to realise Dictable is not pandas. The objects inside Dictable are not meant to be simply primitive types, but actually full objects. The keys are not meant to be columns, they are meant to be variable names. Dictable is more of a programming environment rather than a DataFrame. 

## RDDict (RDD of Dict)
RDDDict is an abstraction layer over pyspark that support the same interface as Dictable allowing us to move from local calculations using Dictable to Spark calculation using pyspark and RDDict without any code-change.

## Dictree (A tree is a dict of dicts)
This class provides us with an ability to be declarative about our tree structures. In its heart, 
```
students:
  id01:
    name: James
    surname: Maxwell
    classes:
        maths: 99
        physics: 95
  id02:
    name: Adam
    surname: Smith
    classes:
        maths: 82
        economics: 97
```
is the idea that we can access elemets declaratively using:
```tree['students/%id/classes/%subject/%grade']``` which can be interpreted to access a Dictable with this values:
```[dict(id = 'id01', subject = 'maths', grade= 99), dict(id = 'id01', subject = 'phyics', grade= 95), ...]```

And indeed, this then allows us to support declarative tree construction like this:
```
tree['students/%id/average/%grade'] = tree['students/%id/classes/%subject/%grade'].groupby('id').apply(np.mean, 'grade')
```




