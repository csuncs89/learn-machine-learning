# Chapter 2 - Convex sets

## Definition of line and line segment
```markdown
Suppose x1 and x2 are two different points in R^n,

then the `line` passing through x1 and x2 has the form:
  `x2 + theta * (x1 - x2)`, where theta is in R,
rewrite it as
  `theta * x1 + (1 - theta) * x2`

If we restrict theta to be within 0 and 1, we we get the `line segment`
```

## Definition of affine set
```markdown
If C is a subset of R^n, and 
if the line passing through any two distinct points lies in C,
  (which means that for any x1 and x2 in C, and any theta in R, 
    we have theta * x1 + (1 - theta) * x2 in C.)
    
Then, the set C is affine
```

```markdown
We can generalize the idea from two points to more points by induction.

Suppose C is an affine set, 
for any three distinct points x1, x2, and x3 in C,
and any three values theta1, theta2, and theta3 in R
  that satisfy theta1 + theta2 + theta3 = 1
  
We need to prove that y = theta1 * x1 + theta2 * x2 + theta3 * x3 is also in C

These theta cannot be all 1, so there must exists a theta != 1,
suppose theta3 is not 1, then 
  y
  = (1 - theta3) / (1 - theta3) * (theta1 * x1 + theta2 * x2) + theta3 * x3
  = (1 - theta3) * 
    ( theta1/(1 - theta3) * x1 + theta2/(1 - theta3) * x2 )
    + theta3 * x3
  = (1 - theta3) *
    ( theta1/(theta1 + theta2) * x1 +  theta2/(theta1 + theta2) * x2 )
    + theta3 * x3
  
  Denote theta1/(theta1 + theta2) * x1 +  theta2/(theta1 + theta2) * x2 as y_tmp
  then, y = (1 - theta3) * y_tmp + theta3 * x3
  It is easy to see that y_tmp is in C, then, y is in C
(when theta1 or theta2 is not 1, the proof is similar, skipped)

By induction, we can easily prove that for any number of distinct points in C,
  (suppose len(points) = n)
  and any thetas in R that satisfy len(thetas) = n and sum(thetas) = 1, 
we will have that `theta1 * x1 + theta2 * x2 + ... + theta_n * x_n is in C`

We call this kind of `theta1 * x1 + theta2 * x2 + ... + theta_n * x_n` as
  `affine combination of points {x1, x2, ..., x_n}`
  
And we can remove the restriction of `distinct` points
  by treating the same points as one, and sum the thetas of the same points as a new theta.

So in human language, we have that
  `affine set contains every affine combination of its points`
```

## Affine set can be expressed as a subspace plus an offset
```markdown

```

## Dimension of affine set C
```markdown
Dimension of the affine set C is the dimension of the subspace V = C - x0,
  where x0 is any element of C
```

## The solution set of a system of linear equations is an affine set, and vice versa
```markdown
```

## Definition of affine hull
```markdown
An affine hull is the set of all affine combinations of points in some subset C of R,
denoted as `aff C`

The affine hull is the smallest affine set that contains C
```

## Definition of affine dimension of a set C
```markdown
The affine dimension of a set C is the dimension of its affine hull
```

## Definition of relative interior of a set C
```markdown
The relative interior of a set C is defined as
  {x in C | intersection of B(x, r) and aff_hull(C) is a subset of C for some r > 0}
  where B(x, r) is a ball of radius r and center x 
  
Compare with the definition of interior of a set C
  {x in C | B(x, r) is a subset of C for some r > 0}
We can see that the ball is made smaller into the set of aff_hull(C).
So that the relative interior might be bigger than interior, because the condition is relaxed
```

## Definition of convex set, convex combination, convex hull
```markdown
If we restrict the thetas in the above definitions to be within 0 and 1, 
we will get the definitions.
```

## Definition of cone (nonnegative homogeneous)
```markdown
A set C is called a cone (nonnegative homogeneous),
  if for every x in C, and theta >= 0, we have theta * x is in C
```

## Definition of convex cone, conic combination (nonnegative linear combination), conic hull
```markdown
If we restrict the thetas of affine related definitions to be >= 0,
we will get the definitions.
```

## Definition of hyperplane and halfspace
```markdown
A hyperplane is a set of the the form:
  {x | transpose(a) * x = b}, where a is in R^n, a != 0, and b in R
  
The two halfspaces are:
  {x | transpose(a) * x >= b},
  {x | transpose(a) * x <= b}
  
geometric interpretations:
suppose we have a point x0 in the hyperplane, then
  transpose(a) * x0 = b, then
  transpose(a) * (x - x0) = 0
which means that a is a normal vector, and the inner product can be 
associated with the angle between a and (x - x0)
```
