
# Numerical and Vector Calculus Programs using SymPy, NumPy, and Matplotlib

This repository contains various Python programs for symbolic vector calculus and numerical methods implemented using `SymPy`, `NumPy`, and `Matplotlib`.

## 📌 Table of Contents

1. [Gradient of a Scalar Field](#1-gradient-of-a-scalar-field)
2. [Divergence of a Vector Field](#2-divergence-of-a-vector-field)
3. [Curl of a Vector Field](#3-curl-of-a-vector-field)
4. [Matrix Rank and Dimension](#4-matrix-rank-and-dimension)
5. [Linear Transformation Visualization](#5-linear-transformation-visualization)
6. [Newton-Raphson Method](#6-newton-raphson-method)
7. [Taylor Series Method for ODEs](#7-taylor-series-method-for-odes)
8. [Runge-Kutta Method](#8-runge-kutta-method)

---

## 1. Gradient of a Scalar Field

```python
from sympy.vector import *
from sympy import symbols

N = CoordSys3D('N')
x, y, z = symbols('x y z')
A = N.x**2 * N.y + 2 * N.x * N.z - 4
delop = Del()
display(delop(A))

gradA = gradient(A)
print(f"\n Gradient of {A} is \n")
display(gradA)
```

---

## 2. Divergence of a Vector Field

```python
from sympy.vector import *
from sympy import symbols

N = CoordSys3D('N')
x, y, z = symbols('x y z')
A = N.x**2 * N.y * N.z * N.i + N.y**2 * N.z * N.x * N.j + N.z**2 * N.x * N.y * N.k
delop = Del()

divA = delop.dot(A)
display(divA)
print(f"\n Divergence of {A} is \n")
display(divergence(A))
```

---

## 3. Curl of a Vector Field

```python
from sympy.vector import *
from sympy import symbols

N = CoordSys3D('N')
x, y, z = symbols('x y z')
A = N.x**2 * N.y * N.z * N.i + N.y**2 * N.z * N.x * N.j + N.z**2 * N.x * N.y * N.k
delop = Del()

curlA = delop.cross(A)
display(curlA)
print(f"\n Curl of {A} is \n")
display(curl(A))
```

---

## 4. Matrix Rank and Dimension

```python
import numpy as np

V = np.array([
    [1, 2, 3],
    [2, 3, 1],
    [3, 1, 2]
])

basis = np.linalg.matrix_rank(V)
dimension = V.shape[0]

print("Basis of the matrix:", basis)
print("Dimension of the matrix:", dimension)
```

---

## 5. Linear Transformation Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

V = np.array([[10, 0]])
origin = np.array([[0, 0, 0], [0, 0, 0]])
A = np.matrix([[-1, 0], [0, 1]])
V1 = np.matrix(V)
V2 = A * np.transpose(V1)
V2 = np.array(V2)

plt.quiver(*origin, V[:, 0], V[:, 1], color=['b'], scale=50)
plt.quiver(*origin, V2[0, :], V2[1, :], color=['r'], scale=50)
plt.show()
```

---

## 6. Newton-Raphson Method

```python
from sympy import *

x = Symbol('x')
g = input('Enter the function: ')
f = lambdify(x, g)
dg = diff(g)
df = lambdify(x, dg)

x0 = float(input('Enter the initial approximation: '))
n = int(input('Enter the number of iterations: '))

for i in range(1, n + 1):
    x1 = (x0 - (f(x0) / df(x0)))
    print('Iteration %d\t Root: %.3f\t Function Value: %.3f' % (i, x1, f(x1)))
    x0 = x1
```

---

## 7. Taylor Series Method for ODEs

```python
from numpy import array, zeros
import math
from numpy import *

def taylor(deriv, x, y, xStop, h):
    x_vals = []
    y_vals = []
    x_vals.append(x)
    y_vals.append(y)
    while x < xStop:
        D = deriv(x, y)
        H = 1.0
        for j in range(3):
            H = H * h / (j + 1)
            y = y + D[j] * H
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    return array(x_vals), array(y_vals)

def deriv(x, y):
    D = zeros((4, 1))
    D[0] = [2 * y[0] + 3 * exp(x)]
    D[1] = [4 * y[0] + 9 * exp(x)]
    D[2] = [8 * y[0] + 21 * exp(x)]
    D[3] = [16 * y[0] + 45 * exp(x)]
    return D

x = 0.0
xStop = 0.3
y = array([0.0, 0.0])
h = 0.1

x, y = taylor(deriv, x, y, xStop, h)

print("The required values are:")
for i in range(len(x)):
    print(f"x={x[i]:.2f}, y={y[i,0]:.5f}")
```

---

## 8. Runge-Kutta Method

```python
from sympy import *
import numpy as np

def RungeKutta(g, x0, h, y0, xn):
    x, y = symbols('x y')
    f = lambdify([x, y], g)
    xt = x0 + h
    y_vals = [y0]

    while xt <= xn:
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)
        y1 = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_vals.append(y1)
        x0 = xt
        y0 = y1
        xt = xt + h

    return np.round(y_vals, 2)

RungeKutta('1+(y/x)', 1, 0.2, 2, 2)
```

---
