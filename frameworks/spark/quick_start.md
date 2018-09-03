```bash
pyspark --master local[2]
```

```python
path = 'file:///tmp/tmp.txt'

textFile = sc.textFile(path)
```
