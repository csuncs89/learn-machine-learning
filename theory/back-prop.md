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
        sum_over_k( w(j, k, l) * a(k, l - 1) ) + b(j, l)  (l >= 2)
        sum_over_k( w(j, k, 1) *    x(k)     ) + b(j, 1)  (k = 1)
        
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
