## Gradient descent is the fastest to minimize y
```
gradient = (grad1, grad2) 
delta_y nearly equals 
    grad1 * delta_x1 + grad2 * delta_x2

When length of (delta_x1, delta_x2) is fixed
    We need to prove that
    when (delta_x1, delta_x2) = (k * grad1, k * grad2), 
    delta_y is maximum

When length of (delta_x1, delta_x2) is fixed, we have
    delta_x1^2 + delta_x2^2 is fixed
    We need to prove
    delta_y = grad1 * k * grad1 + grad2 * k * grad2 = 
        k * (grad1^2 + grad2^2) 
    is the maximum value

x1^2 + x2^2 = fixed value
    Maximize delta_y = grad1 * x1 + grad2 * x2 = 
        ||(grad1, grad2)|| * ||(x1, x2)|| * cos(theta)
    where theta is the angle between vector (g1, g2) and 
        (x1, x2)
    In order to maximize delta_y, make theta 0.
    Then x1, x2 = k * grad1, k * grad2
```

## 