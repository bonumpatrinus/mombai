# mombai
Mombai is a collection of utilities to allow fast research &amp; development
The basic structures are:

## date utilities
1) classes ```BusinessDay``` and ```Month``` that allow direct additions to datetime.datetime and implement standard financial date convention, so that expressions like ```date + 3 * month``` actually work.
2) function ```dt``` that is an all-round "accept anything and cast to datetime.datetime"

## dict utilities
```dict_invert, dict_merge, dict_zip``` and ```dict_apply``` are basic dict manipulations that allow us to: reverse key/values, merge two or more dicts together, handling keys overlap. ```dict_zip``` is equivalent to zip() but for a dict and ```dict_apply``` is equivalent to ```pd.DataFrame.apply```

## Dict
Dict inherits from dict and introduces an important concept: we can access function of keys as well as the keys. Themselves. This, simple extension actually has far reaching implications. It allows us to move away from thinking of Dict as a static container of values, to a full 'calculation graph'. Instead of:

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
x = Dict(a=5, b=2)(c = lambda a, b: a+b)(d = lambda c, b: c+b)(e = lambda c, d: c+d)
```
Dict supports some pandas functionality, namely ```mask``` and ```where```.

## Dictable (Dict and a Table)
Dictable inherits from Dict so has this nice calculation framework but is also a column-based table, with each key being a vector of equal length. The code to track additional variables still looks like:
```
x = x(c = lambda a, b: a+b)(d = lambda c, b: c+b)(e = lambda c, d: c+d)
```
but now this is applied per-row allowing us to track multiple experiments at the same time.
Since Dictable is a table, we also support fast ```apply, sort, groupby, pivot_table, merge, where and mask``` with an interface very similar to pandas. It also supports filtering either directly ```table[mask_of_booleans]``` or using the simpler ```inc, exc``` methods (e.g. ```table.exc(key = None)``` will exclude rows where key is None).

It is important to realise Dictable is not pandas. The objects inside Dictable are not meant to be simply primitive types, but actually full objects. The keys are not meant to be columns, they are meant to be variable names. Dictable is more of a programming environment rather than a DataFrame.

### Dictable and Trees (dict of dicts) 
Dictable provides us with an ability to be declarative about our tree structures. Suppose we work with a yaml-tree like this: 
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
        maths: 92
        economics: 97
```
or, once read using etree:
```
tree = dict(students = dict(id01 = dict(name = 'James', surname = 'Maxwell', classes = dict(maths = 99, physics=95)),
                            id02 = dict(name = 'Adam', surname = 'Smith', classes = dict(maths = 92, economics=97))))
```

At the heart of tree access is the idea that we can access elemets within a tree declaratively using a pattern:
```'students/%id/classes/%subject/%grade'``` 

This means that we can instantiate a Dictable with 

```
d = Dictable(data = tree, columns = 'students/%id/classes/%subject/%grade')
result = Dictable(id = ['id01', 'id01', 'id02', 'id02'], 
                  subject = ['maths', 'physics', 'maths', 'economics'], 
                  grades = [95,99,92,97])
assert eq(d, result)
```

Conversely, we can project back to the tree by 

```
tree_of_grades = d['students/%id/classes/%subject/%grade'] # or
tree_of_grades = d.to_tree('students/%id/classes/%subject/%grade')
```

## RDDict (RDD of Dict, tbc)
RDDict is an abstraction layer over pyspark that support the same interface as Dictable allowing us to move from local calculations using Dictable to Spark calculation using pyspark and RDDict without any code-change.

