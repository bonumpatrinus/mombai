import pyspark
from mombai import Dict

def create_rddict(sc = None):
    if sc is None:
        sc = pyspark.SparkContext._active_spark_context
    class RDDict(pyspark.RDD):
        """
        This class is basically an RDD of Dicts
        """
        def __init__(self, data=None, columns = None, **kwargs):
            if isinstance(data, pyspark.RDD):
                pass
            
rdd = sc.parallelize([Dict(a=1,b=2), Dict(a=2,b=3)])
x.collect()
y = sc.parallelize([("a", 2)])
[(x, tuple(map(list, y))) for x, y in sorted(list(x.cogroup(y).collect()))]


[('a', ([1], [2])), ('b', ([4], []))]
                
