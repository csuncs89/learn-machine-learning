```bash
pyspark --master local[2]
```

```python
path = 'file:///tmp/tmp.txt'

textFile = sc.textFile(path)
```

```
filter(self, f) method of pyspark.rdd.RDD instance
    Return a new RDD containing only the elements that satisfy a predicate.

    >>> rdd = sc.parallelize([1, 2, 3, 4, 5])
    >>> rdd.filter(lambda x: x % 2 == 0).collect()
    [2, 4]
    
parallelize(self, c, numSlices=None) method of pyspark.context.SparkContext instance
    Distribute a local Python collection to form an RDD. Using xrange
    is recommended if the input represents a range for performance.

    >>> sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()
    [[0], [2], [3], [4], [6]]
    >>> sc.parallelize(xrange(0, 6, 2), 5).glom().collect()
    [[], [0], [], [2], [4]]
    
glom(self) method of pyspark.rdd.RDD instance
    Return an RDD created by coalescing all elements within each partition
    into a list.

    >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
    >>> sorted(rdd.glom().collect())
    [[1, 2], [3, 4]]

collect(self) method of pyspark.rdd.RDD instance
    Return a list that contains all of the elements in this RDD.
    
map(self, f, preservesPartitioning=False) method of pyspark.rdd.RDD instance
    Return a new RDD by applying a function to each element of this RDD.

    >>> rdd = sc.parallelize(["b", "a", "c"])
    >>> sorted(rdd.map(lambda x: (x, 1)).collect())
    [('a', 1), ('b', 1), ('c', 1)]

reduce(self, f) method of pyspark.rdd.RDD instance
    Reduces the elements of this RDD using the specified commutative and
    associative binary operator. Currently reduces partitions locally.

    >>> from operator import add
    >>> sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
    15
    >>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
    10
    >>> sc.parallelize([]).reduce(add)
    Traceback (most recent call last):
        ...
    ValueError: Can not reduce() empty RDD
    
flatMap(self, f, preservesPartitioning=False) method of pyspark.rdd.RDD instance
    Return a new RDD by first applying a function to all elements of this
    RDD, and then flattening the results.

    >>> rdd = sc.parallelize([2, 3, 4])
    >>> sorted(rdd.flatMap(lambda x: range(1, x)).collect())
    [1, 1, 1, 2, 2, 3]
    >>> sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect())
    [(2, 2), (2, 2), (3, 3), (3, 3), (4, 4), (4, 4)]
    >>> rdd.flatMap(lambda x: range(1, x)).collect()
    [1, 1, 2, 1, 2, 3]

reduceByKey(self, func, numPartitions=None, partitionFunc=<function portable_hash>) method of pyspark.rdd.RDD instance
    Merge the values for each key using an associative reduce function.

    This will also perform the merging locally on each mapper before
    sending results to a reducer, similarly to a "combiner" in MapReduce.

    Output will be partitioned with C{numPartitions} partitions, or
    the default parallelism level if C{numPartitions} is not specified.
    Default partitioner is hash-partition.

    >>> from operator import add
    >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
    >>> sorted(rdd.reduceByKey(add).collect())
    [('a', 2), ('b', 1)]
```

## Caching
```
cache(self) method of pyspark.rdd.PipelinedRDD instance
    Persist this RDD with the default storage level (C{MEMORY_ONLY_SER}).
```
