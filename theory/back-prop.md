## Gradient descent is the fastest to minimize y
[Definition of gradient](https://en.wikipedia.org/wiki/Gradient#Cartesian_coordinates)
```
gradient = (grad1, grad2)
delta_y nearly equals grad1 * delta_x1 + grad2 * delta_x2

When length of (delta_x1, delta_x2) is fixed, we need to prove that
    when (delta_x1, delta_x2) = (k * grad1, k * grad2), delta_y is maximum

When length of (delta_x1, delta_x2) is fixed, we have
    delta_x1^2 + delta_x2^2 is fixed
    We need to prove
    delta_y = grad1 * k * grad1 + grad2 * k * grad2 = k * (grad1^2 + grad2^2) 
    is the maximum value

x1^2 + x2^2 = fixed value
    We need to maximize delta_y 
        delta_y
        = grad1 * x1 + grad2 * x2 
        = ||(grad1, grad2)|| * ||(x1, x2)|| * cos(theta)
        = ||(grad1, grad2)|| * sqrt(x1^2 + x2^2) * cos(theta)
    where theta is the angle between vector (g1, g2) and (x1, x2)
    In order to maximize delta_y, make theta 0.
    Then x1, x2 = k * grad1, k * grad2
```

## Derivation of backpropagation
```
Goal: calculate partial(Cost, w(j, k, l)) and partial(Cost, b(j, l))
```
`partial(Cost, w(j, k, l)`, `partial(Cost, b(j, l))` are values in
    [Definition of gradient](https://en.wikipedia.org/wiki/Gradient#Cartesian_coordinates)

```
Given the following definitions
    x(j): The j-th dimension value of the input vector x
    
    w(j, k, l):
        Weight for the connection from
            the k-th neuron in the (l - 1)-th layer to 
            the j-th neuron in the   l-th     layer
            
    b(j, l):
        Bias of the j-th neuron in the l-th layer
        
    z(j, l):
        sum_over_k0( w(j, k0, l) * a(k0, l - 1) ) + b(j, l)  (l >= 2)
        sum_over_k0( w(j, k0, 1) *    x(k0)     ) + b(j, 1)  (l = 1)
        
    a(j, l):
        Activation of the j-th neuron in the l-th layer
        = sigma(z(j, l))
        
    y(j, L):
        j-th output in L-th layer
        
    Cost_x stands for the cost given input x:
        Cost_x = sum_over_j( (y(j, L) - a(j, L) )^2 ) / 2
        
    error(j, l) stands for:
        partial(Cost_x, z(j, l))
```
```
Because a(j, L) = sigma(z(j, L))
    error(j, L) 
    = partial(Cost_x, z(j, L))
    = partial(Cost_x, a(j, L)) * partial(a(j, L), z(j, L))
    = partial(Cost_x, a(j, L)) * d(a(j, L), z(j, L))
```
See [Chain rule](https://en.wikipedia.org/wiki/Chain_rule#Higher_dimensions) |
[Chain rule special case](https://wikimedia.org/api/rest_v1/media/math/render/svg/3d059d8743b6dc8824e042fa091e84c39d7db49c)

Both `partial(Cost_x, a(j, L))` and `d(a(j, L), z(j, L))` can be easily computed

```
error(j, l) = partial(Cost_x, z(j, l))
error(j, l + 1) = partial(Cost_x, z(j, l + 1))

z(j, l + 1) = sum_over_k0( w(j, k0, l + 1) * a(k0, l) ) + b(j, l + 1)
a(j=k0, l)     = sigma( z(j=k0, l) )

z(j, l + 1) = sum_over_k0( w(j, k0, l + 1) * sigma(z(j=k0, l))) + b(j, l + 1)

suppose l + 1 layer has 2 neurons, l layer has 3 neurons

z(j=1, l + 1) = w(j=1, k=1, l + 1) * sigma(z(k=1, l)) +
                w(j=1, k=2, l + 1) * sigma(z(k=2, l)) +
                w(j=1, k=3, l + 1) * sigma(z(k=3, l))
                
z(j=2, l + 1) = w(j=2, k=1, l + 1) * sigma(z(k=1, l)) +
                w(j=2, k=2, l + 1) * sigma(z(k=2, l)) +
                w(j=2, k=3, l + 1) * sigma(z(k=3, l))

Because 
    z(1, l + 1) depends on z(j, l)
    z(2, l + 1) depends on z(j, l)

So we have error(j=j0, l)
    = partial(Cost_x, z(j=j0, l))
    = partial( Cost_x, z(j=1, l+1)) * partial(z(j=1, l+1), z(j=j0, l)) ) +
      partial( Cost_x, z(j=2, l+1)) * partial(z(j=2, l+1), z(j=j0, l)) )
    = sum([
            error(j, l + 1) * partial( z(j=k0, l + 1), z(j=j0, l) )
            for k0 in range(2)
      ])
    
    because 
        z(j=k0, l+1) = w(j=k0, k=1, l+1) * sigma(z(j=1, l)) +
                       w(j=k0, k=2, l+1) * sigma(z(j=2, l)) +
                       w(j=k0, k=3, l+1) * sigma(z(j=3, l))
        and only sigma(z(j=j0, l)) depends on z(j=j0, l)
    so we have
    partial(z(j=k0, l+1), z(j=j0, l))
        = w(j=k0, k=j0, l+1) * d(sigma(z(j=j0, l), z(j=j0, l))
    
    so we have
    error(j=j0, l)
        = sum([
            error(j, l+1) * w(j=k0, k=j0, l+1) * d( sigma(z(j=j0, l), z(j=j0, l) )
            for k0 in range(2)
        ])
```

```
Because we have
    z(j0, l):
        sum_over_k0( w(j0, k0, l) * a(k0, l - 1) ) + b(j0, l)  (l >= 2)
        sum_over_k0( w(j0, k0, 1) *    x(k0)     ) + b(j0, 1)  (l = 1)
So only z(j0, l) depends on b(j0, l), so by chain rule:
    partial( Cost_x, b(j0, l) )
        = partial( Cost_x, z(j0, l) )
        = error(j0, l)
```

```
Because we have
    z(j=j0, l):
        sum_over_k0( w(j0, k0, l) * a(j=k0, l - 1) ) + b(j0, l)  (l >= 2)
        ( w(j0, k=1, l) * a(j=1, l-1) + b(j=1, l) +
          w(j0, k=2, l) * a(j=2, l-1) + b(j=2, l) + ...)
        sum_over_k0( w(j0, k0, 1) *    x(j=k0)     ) + b(j0, 1)  (l = 1)

So only z(j=j0, l) depends on w(j0, k0, l), so by chain rule:
    partial( Cost_x, w(j0, k0, l) )
        = a(j=k0, l-1) * partial(Cost_x, z(j0, l))
        = a(j=k0, l-1) * error(j0, l)
```
