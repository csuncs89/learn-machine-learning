# Optimal Online Assignment with Forecasts

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

Basic goal:
  Find an allocation of user visits to contract, so that
  1. at most one ad is shown on each user visit (supply constraint)
  2. each contract fulfills its demand (demand constraint)
Additional goal:
  representative allocations
    pursue more types of users allocated to each contract
```

## Example of the graph
```markdown
We can merge the visits of the same type of users
Example of G:
Gender = {M, F, Unknown}, City = {WA, CA, Unknown}, Age = {5, Unknown}
So each user can be represented as (gender, city, age)
There can at most have 3 * 3 * 2 = 18 supply nodes
And there is a supply for each supply node, eg:
  {M, WA, 5}: 500
  {M, WA, Unknown}: 300
  ....
  {Unknown, Unknown, Unknown}: 0 visits
Also we have some demand nodes, eg: 
  {M, WA}: 100
  ...
Then we can construct the matching graph G between supply and demand nodes
```

## Algorithms
```markdown
Algorithm input:
  G_sample: A sample of G
  J: set of guaranteed contracts
  visits: users visits arrive one at a time
    visits[0], ...,visits[num_visits - 1]

Algorithm:

### The simple idea:
online_alloc = {}
for i_visit in range(num_visits):
  select contract j and display ad of it to visits[i_visit] according to
    G_sample, J, {d_j}, visits[:i_visit], online_alloc
  online_alloc.add((i_visit, j))
  
### Idea of this paper:
create an offline allocation plan according to G_sample

for i_visit in range(num_visits):
  select contract j and display ad of it to visits[i_visit] according to
    offline_alloc_plan, J, {d_j}, visits[i_visit]

Space of offline_plan is O(size(J))
```


