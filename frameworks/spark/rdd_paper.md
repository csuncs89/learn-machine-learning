# What is RDD?
An RDD is a read-only, partitioned collection of records

# How can RDD be created?
RDDs can only be created through deterministic operations on either
* data in stable storage
* other RDDs
These operations are called transformations

# What is an `transformation`?
See above

# How to recover from failure?
- RDD has enough information to compute its partitions from data in stable storage
- The information is: how it was derived from other datasets or stable storage
(in a word: recover using lineage)

# What is an `action`?
Operation that 
- returns a value to the application or 
- exports data to a storage system
