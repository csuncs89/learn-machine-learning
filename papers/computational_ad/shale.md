# SHALE: An Efficient Algorithm for Allocation of Guaranteed Display Advertising

## Problem description
[A short version](opt_online_assign_w_f.md#problem-description)
```markdown
I: set of individual user visits (supply nodes)
J: set of guaranteed contracts (demand nodes)
E: set of edges between I and J
  there is an edge connecting user i and contract j if 
    user i matches the targeting constraints of contract j
     
s[i]: supply of user i
  representing how many times the user visits during the time period
d[j]: demand of contract j

G: (vertices=union(I, J), edges=E), it is a bipartite graph
S[j]: total eligible supply for contract j

x(i, j): fraction(probability) of s[i] allocated to d[j]
  sum(x(i, j), for contract j in neighbor(i)) == 1
         
u[j]: under-delivery of contract j
  i.e. number of user visits delivered less than d[j]
  if demand constraint is satisfied, we will have
    u[j] >= d[j] - sum(s[i] * x(i, j), for user i in neighbor(j))
p[j]: penalty of contract j for under-delivery
  larger values mean that under-delivery will be penalized more
        
theta(i, j): 
  demand of contract j / total eligible supply for contract j
  this is the ideal allocation that allocates every eligible supply
    with equal probability
  i.e. d[j] / S[j], where S[j] = sum(s[i], for i in neighbor(j))
  ( eligible means there is an edge
    example:
    s[1] (10), s[2] (20), s[5] (70) are eligible supply for d[1] (12),
      then we have theta(1, 1), theta(2, 1), theta(5, 1) all equal to 0.12
      obviously if we choose x(1, 1), x(2, 1), x(5, 1) all as 0.12,
      the demand can be satisfied, and this is the ideal allocation that
      allocates every eligible supply with equal probability)

non-representativeness for contract j:
  f_non_repr(j) = 
    1/2 * sum( s[i] * V[j] / theta(i, j) * 
                 (x(i, j) - theta(i, j)) ** 2,
               for i in neighbor(j)
          )
  where
  V[j]: 
    the relative priority of the contract j
    a larger V[j] means that this contract's representativeness 
      is more important
  s[i]:
    the larger the supply, the more important of its representativeness,
      which will make larger supply's x(i, j) closer to theta(i, j)
      besides, adding s[i] will make the function scale-free
      (see paper: Optimal Online Assignment with Forecasts)

## Goal
Minimize
  1/2 * sum(f_non_repr(j), for j in all contracts) + 
    sum(p[j] * u[j], for j in all contracts)
    
  which means "jointly minimize the distance between the allocation and its ideal one,
    and the penalty of under-delivery"
    
s.t.
  1. demand constraints are satisfied:
    for all j, we have
      u[j] >= d[j] - sum(s[i] * x(i, j), for user i in neighbor(j))
  2. supply constrains are satisfied:
    for all i, we have
      sum(x(i, j), for j in neighbor(i)) <= 1
  3. non-negativity constraints are satisfied
    for all i, j, we have
      x(i, j) >= 0
      u[j] >= 0
```

## Online serving with forecasts (forecast: sample of G)
- Offline phase
  - input: a sample of graph G, G_sample, with nodes: I_sample, J_sample
  - create an allocation plan with space complexity O(J_sample)
  
- Online phase
  - input
    - the offline allocation plan
    - user visits one by one
    - all the contracts
  - decide which contract to serve to the user visit

## Algorithms
- Standard algorithm
- HWM algorithm
- SHALE algorithm, which is a combination of the above two 

## Standard algorithm
- Offline phase
```markdown
Solve the problem using standard methods (dual problem, KKT)
Output: demand duals alpha, alpha(j) is for contract j
```

- Online phase
```markdown
User i arrives
Find the value beta(i) by solving the equation
  sum( g(i, j, alpha(j) - beta(i)),
       for j in neighbor(i) ) = 1
  where
    g(i, j, z) = max{0, theta(i, j) * (1 + z/V(j))}
x(i, j) = g(i, j, alpha(j) - beta(i))
```

## HWM algorithm
HWM: High Water Mark

The short version
```markdown
Order all contracts by their allocation order 
For contract in the ordered contracts:
    Try to allocate an equal fraction from all the eligible supply
```

The long version
```markdown
order all demand nodes in decreasing contention order (d[j]/S[j])

for each supply node i:
    s_remain[i] = s[i]
    
for j in allocation order:
    find fraction such that 
        sum( min(s_remain[i], fraction * s[i]), 
             for i in neighbor(j) ) = d[j]
    if fraction does not exist:
        fraction = float('inf')
    for i in neighbor(j):
        s_remain[i] -= min(s_remain[i], fraction * s[i])
```

## KKT
See [Lagrange Multipliers and the Karush-Kuhn-Tucker conditions](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf)

## Key innovation of SHALE
- The ability to take any dual solution and convert it into a good primal solution
- Achieve this by extending the simple heuristic HWM to incorporate dual values

## Tow pieces of SHALE
- Piece 1: Find resonable duals, which is an iterative algorithm
- Piece 2: Convert the reasonable set of duals into a good primal solution

## SHALE algorithm
```markdown

```

## Proof of SHALE
```markdown
d(j, alpha) = sum(s(i) * g(i, j, alpha(j) - beta(i)),
                  for i in neighbor(j))
                  
d(j) - d(j, alpha(step=t)) <= d(j) * (alpha(step=t+1, j) - alpha(step=t, j) / V(j)
V(j) * (d(j) - d(j, alpha(step=t)) <= d(j) * (alpha(step=t+1, j) - alpha(step=t, j))
V(j) * (1 - d(j, alpha(step=t) / d(j)) <= alpha(step=t+1, j) - alpha(step=t, j)
V(j) * (1 - d(j, alpha(step=t) / d(j)) + alpha(step=t, j) <= alpha(step=t+1, j)

g(i, j, alpha(j) - beta(i)) = max{0, theta(i, j) * (1 + (alpha(j) - beta(i))/V(j) )}
sum(max{0, 
        theta(i, j) * 
        (1 + (alpha(step=t, j) - beta(step=t, i)) / V(j) ) 
    },
    for j in neighbor(i) 
) = 1

max{0, theta(i, j=1) * (1 + (alpha(step=t, j=1) - beta(step=t, i)) / V(j) ) } +
max{0, theta(i, j=3) * (1 + (alpha(step=t, j=3) - beta(step=t, i)) / V(j) ) } +
... = 1

alpha(step=t+1, j) >= alpha(step=t, j) 

max{0, theta(i, j=1) * (1 + (alpha(step=t+1, j=1) - beta(step=t, i)) / V(j) ) } +
max{0, theta(i, j=3) * (1 + (alpha(step=t+1, j=3) - beta(step=t, i)) / V(j) ) } +
... >= 1

So in order to make it equal 1, we have to increase beta(step=t, i) as beta(step=t+1, i)
```
