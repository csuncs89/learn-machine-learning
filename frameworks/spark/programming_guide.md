ref: https://spark.apache.org/docs/1.6.0/programming-guide.html#spark-programming-guide

# What is a driver program?
- https://spark.apache.org/docs/1.6.0/cluster-overview.html
- The process running the main() function of the application and creating the SparkContext

# What is a SparkContext?
- https://spark.apache.org/docs/1.6.0/api/python/pyspark.html?highlight=sparkcontext#pyspark.SparkContext
- Main entry point for Spark functionality. 
  - A SparkContext represents the connection to a Spark cluster
  - can be used to create RDD and broadcast variables on that cluster.
  
# What is RDD?
- https://spark.apache.org/docs/1.6.0/api/python/pyspark.html#pyspark.RDD
- A Resilient Distributed Dataset (RDD), the basic abstraction in Spark
- Represents an immutable, partitioned collection of elements that can be operated on in parallel.
  - immutable
  - partitioned
  - can be operated in parallel

# What are the two types of shared variables?
- broadcast variables
  - https://spark.apache.org/docs/1.6.0/api/python/pyspark.html?highlight=broadcast#pyspark.Broadcast
  - can be used to cache  value in memory on all nodes
- accumulators
  - https://spark.apache.org/docs/1.6.0/api/python/pyspark.html?highlight=broadcast#pyspark.Accumulator
  
