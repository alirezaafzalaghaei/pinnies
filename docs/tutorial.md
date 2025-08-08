# PINNIES Tutorial

Welcome to the PINNIES tutorial! This guide will walk you through solving various types of integral equations using the PINNIES library. We'll cover Fredholm and Volterra equations in 1D, 2D, and 3D, as well as a fractional integro-differential equation example.

PINNIES leverages the power of Physics-Informed Neural Networks (PINNs) to approximate solutions to these complex mathematical problems. Instead of relying on traditional numerical methods, PINNIES uses neural networks to learn the solution that satisfies the given integral equation.

## Getting Started

First, let's import the necessary libraries. We'll need `numpy` for numerical operations, `torch` for building our neural networks, and of course, the components from the `pinnies` library itself.

```python
import numpy as np
import torch
from pinnies import *
from torch import nn
```

---

## 1. Fredholm Integral Equation of the First Kind (1D)

Let's start with a classic example: a 1D Fredholm integral equation of the first kind. The general form of this equation is:

$$
f(x) = \int_a^b K(x, t) y(t) dt
$$

In this specific problem, we aim to solve:

$$
y(x) - \int_0^1 t y(t) dt = e^x + x - \frac{4}{3}
$$

The goal is to find the unknown function $y(x)$. For this particular problem, an exact solution is known to be $y(x) = x + e^x$, which we'll use for validation.

### Implementation

To solve this with PINNIES, we define a Python class that inherits from `pinnies.Fredholm1D`. This class encapsulates the specific details of our problem.

```python
class Problem(Fredholm1D):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain
        self.model = model
        self.F = torch.exp(self.x) + self.x - 4 / 3
        self.K = self.T

    def residual(self):
        y = self.predict(self.x)
        zeta = y
        I = self.quad(self.K.T * zeta, self.a, self.b)
        return [y - I - self.F]

    def exact(self, x):
        # For validation
        return x + torch.exp(x)
```

Let's break down what's happening in this class:

-   **`__init__(self, domain, num_train, model)`**: The constructor initializes the parent `Fredholm1D` class and sets up the problem-specific attributes.
    -   `domain`: A list or tuple `[a, b]` defining the integration interval.
    -   `num_train`: The number of training points to use within the domain.
    -   `model`: The neural network model that will approximate the solution $y(x)$.
    -   `self.F`: This represents the known right-hand side of the equation.
    -   `self.K`: This is the kernel of the integral, $K(x, t)$. In this case, it's simply $t$.

-   **`residual(self)`**: This is the core of the PINN approach. The residual is the part of the equation that should be equal to zero. By minimizing the residual, we train the neural network to find the solution. Here, the residual is `y - I - self.F`, where `I` is the integral term computed using the `self.quad` method.

-   **`exact(self, x)`**: This optional method provides the analytical solution. PINNIES can use this to calculate the validation error during training, which helps in monitoring the model's performance.

### Training the Model

Now that we have defined our problem, we need to set up a neural network and train it.

```python
set_seed(42)

model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p = Problem([0, 1], 30, model)
p.solve(10)
```

Here's what this code does:

1.  `set_seed(42)`: We set a random seed for reproducibility.
2.  `model = nn.Sequential(...)`: We define a simple feed-forward neural network using `torch.nn`. This network will take $x$ as input and output the corresponding value of $y(x)$.
3.  `p = Problem([0, 1], 30, model)`: We create an instance of our `Problem` class.
4.  `p.solve(10)`: We call the `solve` method to start the training process for 10 epochs.

The `solve` method uses an optimizer to adjust the weights of the neural network to minimize the residual defined in the `residual` method. The output will show the training loss and validation error decreasing over epochs.

---

## 2. Volterra Integral Equation of the First Kind (1D)

Next, we'll tackle a 1D Volterra integral equation. These are distinct from Fredholm equations in that the upper limit of integration is a variable, $x$. The general form is:

$$
y'(x) + y(x) - \int_a^x K(x, t) y(t) dt = 0
$$

For this example, we solve:

$$
y'(x) + y(x) - \int_0^x e^{t-x} y(t) dt = 0
$$

with the initial condition $y(0) = 1$. The exact solution is $y(x) = e^{-x} \cosh(x)$.

### Implementation

The implementation is similar to the Fredholm case, but we inherit from `pinnies.Volterra1D` and need to handle the initial condition.

```python
class Problem(Volterra1D):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain
        self.model = model
        self.K = torch.exp(self.T - self.X)
        self.zero = torch.tensor([[0.0]])

    def residual(self):
        y = self.predict(self.x)
        y_x = self.diff(y, self.x, n=1)

        u_t = self.predict_on_T()
        zeta = u_t
        I = self.quad(self.K * zeta, self.a, self.x)

        initial = self.get_initial()
        return [y_x + y - I, initial]

    def get_initial(self):
        return self.predict(self.zero) - (1)

    def exact(self, x):
        # For validation
        return torch.exp(-x) * torch.cosh(x)
```

Key differences from the Fredholm example:

-   **`residual(self)`**: The residual list now contains two components: the main equation and the initial condition. This ensures that the model learns to satisfy both. The `self.diff` method is used to compute the derivative $y'(x)$.
-   **`get_initial(self)`**: This helper method calculates the value of our predicted function at $x=0$ and subtracts the known initial value. The result is used as part of the residual.

### Training the Model

We define a slightly deeper model and then instantiate and solve the problem.

```python
set_seed(42)

model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p2 = Problem((0, 5), 10, model)
p2.solve(10, learning_rate=0.3)
```

The training process is the same, but we've specified a custom `learning_rate` for the optimizer.

---

## 3. Fredholm Integral Equation (2D)

Now let's move to higher dimensions. Here is a 2D Fredholm equation:

$$
y(x, y) - \int_a^b \int_c^d K(x, y, s, t) y(s, t) ds dt = f(x, y)
$$

Specifically, we solve:

$$
y(x, y) - \int_0^2 \int_0^1 (-\frac{1}{2}xt) y(s, t) ds dt = x^2y + \frac{4}{9}x
$$

The exact solution is $y(x, y) = x^2y$.

### Implementation

We inherit from `pinnies.Fredholm2D`, and the logic extends naturally from the 1D case.

```python
class Problem(Fredholm2D):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain[0]
        self.c, self.d = domain[1]
        self.model = mlp
        source = lambda x, y: (x**2 * y) + ((4 / 9) * x)
        self.F = source(self.x[:, 0], self.x[:, 1]).reshape(self.N, self.N)
        self.K = -(1 / 2 * self.X * self.T)

    def residual(self):
        y = self.predict(self.x).reshape(self.N, self.N)
        zeta = y

        I = self.quad(self.K * zeta, self.c, self.d)
        I = self.quad(I, self.a, self.b)
        return [y - I - self.F]

    def exact(self, X):
        # For validation
        x, y = X[:, 0], X[:, 1]
        return x**2 * y
```

The main difference is that our domain is now 2D, and the inputs to our model are 2D coordinates `(x, y)`. The integration is performed iteratively over the two dimensions.

### Training the Model

The model now takes a 2D input and outputs a single value.

```python
set_seed(42)

mlp = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p1 = Problem([(0, 1), (0, 2)], 25, mlp)
p1.solve(100)
```

---

## 4. Volterra Integral Equation (2D)

A 2D Volterra equation has variable upper limits for both integrals:

$$
y(x, y) + \int_0^y \int_0^x K(x, y, s, t) y(s, t) ds dt = f(x, y)
$$

We will solve:

$$
y(x, y) + \int_0^y \int_0^x e^{(s+t)} e^{(x+y)} y(s, t) ds dt = f(x, y)
$$

where $f(x, y)$ is a more complex source term and the exact solution is $y(x, y) = x+y$.

### Implementation

```python
class Problem(Volterra2D):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain[0]
        self.c, self.d = domain[1]
        self.model = mlp
        source = lambda x, y: (
            x
            + y
            + torch.exp(x + y)
            * (
                x * torch.exp(x + y)
                + y * torch.exp(x + y)
                - 2 * torch.exp(x + y)
                - torch.exp(x) * x
                - torch.exp(y) * y
                + 2 * torch.exp(x)
                + 2 * torch.exp(y)
                - 2
            )
        )
        self.F = source(self.x[:, 0], self.x[:, 1]).reshape(self.N, self.N)
        self.K = torch.exp(self.S + self.T) * torch.exp(self.X + self.Y)

    def residual(self):
        y = self.predict(self.x).reshape(self.N, self.N)
        zeta = self.predict_on_ST()

        I = self.quad(self.K * zeta, self.c, self.bdy)

        I = self.quad(I.permute(2, 1, 0), self.a, self.bdx.reshape(1, -1))

        return [y + I - self.F]

    def exact(self, X):
        # For validation
        x, y = X[:, 0], X[:, 1]
        return x + y
```

The implementation follows the same pattern, but the integration is more complex due to the variable upper limits.

### Training

```python
set_seed(42)

mlp = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p1 = Problem([(0, 1), (0, 2)], 15, mlp)
p1.solve(20)
```

---

## 5. Fredholm Integral Equation (3D)

The principles extend to 3D. Here's a 3D Fredholm equation:

$$
y(x, y, z) - \int_e^f \int_c^d \int_a^b K(x, y, z, r, s, t) y(r, s, t) dr ds dt = f(x, y, z)
$$

We solve:

$$
y(x, y, z) - \int_1^2 \int_{-1}^1 \int_0^1 (e^s r) y(r, s, t) dr ds dt = x^2 y e^z - \frac{e^2 - e^{-1}}{2}
$$

The exact solution is $y(x, y, z) = x^2 y e^z$.

### Implementation

```python
class Problem(Fredholm3D):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain[0]
        self.c, self.d = domain[1]
        self.e, self.f = domain[2]

        self.model = mlp
        source = lambda x, y, z: (
            x**2 * y * torch.exp(z) - ((-torch.e + np.exp(2)) * np.exp(-1)) / 2
        )
        self.F = source(self.x[:, 0], self.x[:, 1], self.x[:, 2]).reshape(
            self.N, self.N, self.N
        )
        self.K = torch.exp(self.S) * self.R

    def residual(self):
        y = self.predict(self.x).reshape(self.N, self.N, self.N)
        zeta = y
        I = self.quad(self.K * zeta, self.e, self.f)
        I = self.quad(I, self.c, self.d)
        I = self.quad(I, self.a, self.b)
        return [y - I - self.F]

    def exact(self, X):
        # For validation
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return x**2 * y * torch.exp(z)
```

### Training

The model now takes a 3D input.

```python
set_seed(42)

mlp = nn.Sequential(
    nn.Linear(3, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p1 = Problem([(0, 1), (-1, 1), (1, 2)], 10, mlp)
p1.solve(10)
```

---

## 6. Fractional Integro-Differential Equation

Finally, PINNIES can handle more exotic equations, like this fractional integro-differential equation. This type of equation involves both integrals and fractional derivatives.

$$
\kappa D^{0.5}y(x) - y(x) + y^2(x) + y(x)\int_0^x y(t)dt = 0
$$

with initial condition $y(0) = u_0$. Here $D^{0.5}$ is the Caputo fractional derivative of order 0.5.

### Implementation

We can combine classes from PINNIES to build the problem. Here we use `Volterra1D` for the integral part and `Fractional` for the fractional derivative. This demonstrates the modularity of the library.

```python
class Problem(Volterra1D, Fractional):
    def __init__(self, domain, num_train, model):
        super().__init__(domain, num_train)
        self.a, self.b = domain
        self.model = model
        self.kappa = 2
        self.u0 = 0.1
        self.K = torch.ones_like(self.T)
        self.M = self.fracmatrix(0.5)

    def residual(self):
        y = self.predict(self.x)
        y_x = self.diff(y, self.x, n=1)

        u_t = self.predict_on_T()
        zeta = u_t
        I = self.quad(self.K * zeta, self.a, self.x)

        initial = self.get_initial()
        return [self.kappa * (self.M @ y) - y + y**2 + y * I, 1e1 * initial]

    def get_initial(self):
        zero = torch.tensor([[0.0]])
        return self.predict(zero) - self.u0

    def criteria(self):
        x_test = torch.linspace(self.a, self.b, 10000).reshape(-1, 1)
        predict = (self.predict(x_test)).detach()
        x_test = x_test.detach().numpy()
        predict = predict.detach().numpy()

        u_max_pred = np.max(predict)
        u_max_exact = 1 + self.kappa * np.log(self.kappa / (1 + self.kappa - self.u0))

        return np.abs(u_max_exact - u_max_pred)
```

-   **`__init__`**: We define constants `kappa` and `u0`. `self.M` is the matrix representation of the fractional derivative operator, obtained via `self.fracmatrix(0.5)`.
-   **`residual`**: The residual combines the fractional derivative term (`self.M @ y`), the integral term, the non-linear terms, and the initial condition. Note that the initial condition is multiplied by `1e1` to give it a higher weight in the loss function.
-   **`criteria`**: This method defines a custom metric for this problem, as a simple exact solution is not available. It compares the maximum value of the predicted solution to a known property of the exact solution.

### Training

```python
set_seed(42)

model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)

p2 = Problem((0, 5), 20, model)

p2.solve(10, learning_rate=0.1)

print(p2.criteria())
```

After training, we call the `criteria` method to evaluate our custom metric.

---

## Conclusion

This tutorial has demonstrated how to use PINNIES to solve a variety of integral and integro-differential equations. The key steps are:

1.  **Define a problem class**: Inherit from the appropriate base class (e.g., `Fredholm1D`, `Volterra2D`).
2.  **Implement the `residual` method**: This is where you define the equation(s) you want to solve.
3.  **Define a neural network**: Use `torch.nn` to create a model that will approximate the solution.
4.  **Instantiate and solve**: Create an instance of your problem class and call the `solve` method.

PINNIES is a flexible library that allows you to tackle complex mathematical problems with the power of neural networks. Feel free to experiment with different equations, network architectures, and training parameters to explore its full capabilities.
