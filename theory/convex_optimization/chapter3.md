# Chapter 3 - Convex functions

## Definition of convex function
```markdown
A function f: R^n -> R is convex
  if domain(f) is a convex set and
  if for all x, y in domain(f), and theta in [0, 1], we have
    f(theta * x + (1 - theta) * y) <= theta * f(x) + (1 - theta) * f(y)
```
- Geometrically, the line segment between (x, f(x)) and (y, f(y)) is above the graph of f. 
  - According to [Definition of line and line segment](chapter2.md#definition-of-line-and-line-segment),  
  - a point in the line segment between (x, f(x)) and (y, f(y)) is  
  - `( theta * x + (1 - theta) * y, theta * f(x) + (1 - theta) * f(y) )`  
  - a corresponding (the same x-axis) point in the graph is  
  -  `( theta * x + (1 - theta) * y, f(theta*x + (1-theta)*y )`
  
## Definition of epigraph
```markdown
The graph of a function f: R^n -> R is defined as
  { (x, f(x)) | x is in domain(f) }
which is a subset of R^(n+1).

The epigraph of a function f: R^n -> R is defined as
  epi(f) = { (x, t) | x is in domain(f), f(x) <= t }
which is also a subset of R^(n+1). Epigraph means 'above the graph'.
```

## A function is convex if and only if its epigraph is a convex set
```markdown
```

## Definition of pointwise maximum
```markdown
Pointwise maximum of f1 and f2 is defined as:
  f = max{f1(x), f2(x)}
```

## Pointwise maximum preserves convexity
```markdown
```
