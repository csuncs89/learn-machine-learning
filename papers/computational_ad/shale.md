# SHALE: An Efficient Algorithm for Allocation of Guaranteed Display Advertising

## Problem description
```markdown
I: set of individual user visits
J: set of guaranteed contracts
E: set of edge between I and J
  there is an edge (i, j)
  if user i matches the targeting constraints of contract j
d_j: demand of contract j
s_i: supply of user i
  representing how many times the user visits during the time period
G: (union(I, J), E)

x(i, j): fraction(probability) of s(i) allocated to d_j
  sum(x(i, j), for contract j in neighbor(i)) == 1
p_j: penalty of contract j
u_j: under-delivery of contract j
  i.e. number of user visits delivered less than d_j
  d_j - sum(s_i * x(i, j), for user i in neighbor(j))
theta(i, j): demand of contract j / total eligible supply for contract j
  d_j / S_j, where S_j = sum(s_i, for i in neighbor(j))
  ( eligible means there is a connected edge
    example:
      s_1 (10), s_2 (20), s_5 (70) are eligible supply for d_1 (12),
      then we have s(1, 1), s(2, 1), s(5, 1) all equal to 0.12
      obvious if we choose x(1, 1), x(2, 1), x(5, 1) all as 0.12,
      the demand can be satisfied )
non-representativeness for contract j:
  1/2 * sum( s_i * V_j / theta(i, j) *  (x(i, j) - theta(i, j)) ** 2,
             for i in neighbor(j)
        )
  where
  V_j: the relative priority of the contract j
    a larger V_j means that this contract's representativeness is more important
  s_i: the larger the supply, the more important of its representativeness,
    which will make larger supply's x(i, j) closer to theta(i, j)
    besides, adding s_i will make the function scale-free
    (see paper: Optimal Online Assignment with Forecasts)

```
