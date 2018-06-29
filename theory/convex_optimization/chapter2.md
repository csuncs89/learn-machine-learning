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

## Definition of polyhedra
```markdown
Solution set of a finite number of finite number of linear equalities
and inequalities
Thus it is the intersection of a finite number of halfspaces and hyperplanes

Polyheras are convex sets.
```

## Definition of simplex
```markdown
convex hull of k + 1 afflinely independent points (v0, ..., vk) in R^n
(afflinely independent means that v1 - v0, ..., vk - v0 are linearly independent)
```

## Simplex can be described as a polyhedra
```markdown
Suppose C is a simplex, then
  x in C, if and only if x = theta0 * v0 + ... + theta_k * v_k, for some thetas with
    any theta >= 0 and sum(thetas) = 1
  then
  x = (1 - theta1 - ... - theta_k) * v0 + theta1 * v1 + ... + theta_k * v_k
  x = v0 + theta1 * (v1 - v0) + ... + theta_k * (v_k - v0)
  denote (theta1, ..., theta_k) as y, then
  any theta in y >= 0 and sum(y) <= 1
so if we denote [ v1 - v0, ..., v_k - v0 ] as B (shape is n*k)
then the original definition is equivalent to 
  x in C, if and only if
    x = v0 + B * y for some y with any theta >= 0 and sum(y) <= 1
the affine independence of v0 to v_k implies that the matrix B has rank k.
Therefore there exists a nonsingular matrix A = (A1, A2) in R^(n*n), such that
  AB = stack_rows(A1, A2) * B = stack_rows(I, 0) (shape is n*k)
  
  multiply A to x = v0 + B * y,
  so stack_rows(A1, A2) * x = stack_rows(A1, A2) * (v0 + B * y)
  which is equaivalent to both
  `A1 * x = A1 * (v0 + B * y)` and `A2 * x = A2 * (v0 + B * y)` are satisfied, then we get
  `A1 * x = A1 * v0 + y` and `A2 * x = A2 * v0`, and by the condition above
    any theta in y >= 0 and sum(y) <= 1, we can get a set of linear equalities and inequilities 
  
  `A1 * x - A1 * v0 >= 0` and
  `transpose(1) * (A1 * x - A1 * v0) <= 1` and
  `A2 * x = A2 * v0`
  
So simplex can be described as a ployhedra
```

## Operations that preserve convexity
```markdown
intersection of convex sets is convex

projection of convex set onto some of its coordinates is convex

sum of two convex sets is convex

partial sum of S1, S2, S1 is in R^n x R^m and S2 is in R^n x R^m,
  defined as S = {(x, y1 + y2) | (x, y1) in S1, (x, y2) in S2}
  if both S1 and S2 are convex, then partial sum of S1 and S2 is convex
  
affine function 
  f(x) = Ax + b that transforms from R^n to R^m, and A is in R^(m*n) and b is in R^m
if S is subset of R^n and S is convex, then the image of S under f is convex
and the inverse image of S under f is also convex
  (inverse_f(S) = {x | f(x) in S})
  
perspective function
  the perspective function scales or normalizes vectors so that the last component
  is one, and then drops the last component
  (formally: 
    define the perspective function P: R^(n+1) -> R^n, 
      with domain_P = decartes(R^n, set_of_positive_numbers).
      as P(z, t) = z/t 
   )
if a subset C of domain_P is convex, then its image P(C) = {P(x) | x in C} is convex
  The result is intuitive: a convex object, viewed through a pin-hole camera, yields a convex image
The inverse image of a convex set under the perspective function is also convex

linear-fractional function
  it is P(g(x)) where g is an affine function and P is a perspective function
linear-fractional function preserve convexity
```

## Definition of minimum and minimal
- https://www.zhihu.com/question/22319675
- minimum: 最小值，minimal：极小值

## Separating hyperplane therom
```markdown
Suppose C and D are nonempty disjoint convex sets, then there exists a != 0 and b
  such that transpose(a) * x <= b for all x in C and 
  transpose(a) * x >= b for all x in D
```

## Definition of closure, boundary and interior of a set
- closure: https://en.wikipedia.org/wiki/Closure_(topology)
- boundary: https://en.wikipedia.org/wiki/Boundary_(topology)
- interior: https://en.wikipedia.org/wiki/Interior_(topology)

## Definition of supporting hyperplane
```markdown
Suppose C is a subset of R^n, and x0 is a point in its boundary,
  if there exists a != 0 such that 
  transpose(a) * x <= transpose(a) * x0 is true for all x in C, then
  the hyperplane {x | tranpose(a) * x = tranpose(a) * x} is called
  a supporting hyperplane to C at the point x0
```

## Supporting hyperplane therom
```markdown
For any nonempty convex set C, and for any x0 in its boundary, there
  exists a supporting hyperplane to C at x0
```

## Definition of dual cone
```markdown
K is a cone, the dual cone is defined as
  dual_cone(K) = { y | tranpose(x) * y >= 0 for all x in K }
dual_cone(K) is a cone and is always convex
```
https://en.wikipedia.org/wiki/Dual_cone_and_polar_cone
