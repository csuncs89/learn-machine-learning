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

Suppose C is an affine set, for any three distinct points x1, x2, x3, we have
  `x_tmp = theta1 * x1 + theta2 * x2 is in R^n provided that theta1 + theta2 = 1`
If x_tmp is the same as x3, we will have 
  `theta1 * x1 + theta2 * x2 + theta3 * x3 is in R^n, where theta3 is 0.`
  so we have theta1 + theta2 + theta3 = 0
Elif x_tmp is not the same as x3, we will have
  `theta3 * (theta1 * x1 + theta2 * x2) + (1 - theta3) * x3 is in R^n`
    which is
  `theta1 * theta3 * x1 + theta2 * theta3 * x2 + (1 - theta3) * x3`,
  so we have theta1 * theta3 + theta2 * theta3 + (1 - theta3)
   = (theta1 + theta2) * theta3 + (1 - theta3)
   = 1 * theta3 + (1 - theta3)
   = 1
```
