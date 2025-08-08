import torch
from scipy.special import gamma, roots_jacobi


class Fractional:
    def fracmatrix(self, alpha: float) -> torch.Tensor:
        """
        Compute the fractional integration matrix based on the given order alpha.

        Args:
            alpha (float): The order of the fractional derivative.

        Returns:
            torch.Tensor: The fractional integration matrix.
        """
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        b = self.b
        N = len(self.x)
        A = torch.zeros((N, N))
        t = self.x

        for i in range(1, N):
            A[i, : i + 1] = self.fracweights(t[: i + 1], alpha)
        return A

    def fracweights(self, t: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute the weights for fractional integration.

        Args:
            t (torch.Tensor): The time vector.
            alpha (float): The order of the fractional derivative.

        Returns:
            torch.Tensor: The computed weights.
        """
        n = len(t)
        mu = torch.zeros(n + 1)
        for k in range(n - 1):
            mu[k + 1] = (
                (t[-1] - t[k]) ** (1 - alpha) - (t[-1] - t[k + 1]) ** (1 - alpha)
            ) / ((t[k + 1] - t[k]))
        w = (mu[:-1] - mu[1:]) / gamma(2 - alpha)
        return w



class FractionalGJ:
    """
    Computes the Caputo fractional derivative using Gauss-Jacobi quadrature.

    This class structure closely follows the user-provided NumPy version,
    but is implemented in PyTorch to support automatic differentiation.

    It is designed to be a mix-in for a problem class that provides a `diff` method.
    """

    def __init__(self, alpha: float, n_quad: int = 100):
        """
        Initializes the fractional operator.

        Args:
            alpha (float): The order of the fractional derivative (0 < alpha < 1).
            n_quad (int): The number of quadrature points.
        """
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")

        self.alpha = alpha
        self.n_quad = n_quad

        # Pre-compute nodes and weights for efficiency. They are constant for a given
        # alpha and n_quad.
        nodes, weights = roots_jacobi(self.n_quad, -self.alpha, 0)

        # We will convert these to PyTorch tensors on the correct device when needed.
        self._numpy_u = nodes
        self._numpy_w = weights
        self.u, self.w = None, None  # To be initialized on the correct device later

    def fracdiff(self, func: callable, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Caputo derivative of a function `func` at points `x`.
        This is the main public method, analogous to `fracdiff`.

        Args:
            func (callable): The function to differentiate (e.g., a model's forward pass).
            x (torch.Tensor): The points at which to evaluate the derivative.

        Returns:
            torch.Tensor: The computed derivative at each point in x.
        """
        # Ensure nodes and weights are PyTorch tensors on the same device as the input.
        if self.u is None or self.u.device != x.device:
            self.u = torch.tensor(self._numpy_u, dtype=x.dtype, device=x.device)
            self.w = torch.tensor(self._numpy_w, dtype=x.dtype, device=x.device)

        derivatives = []
        # Loop is necessary because the integration domain [0, x_i] changes for each point.
        for x_point in x.flatten():
            derivatives.append(self._caputo_op_single(func, x_point))

        # Stack the results into a single tensor for use in loss computation.
        return torch.stack(derivatives).reshape(-1, 1)

    def _caputo_op_single(self, func: callable, x_point: torch.Tensor) -> torch.Tensor:
        """
        Computes the Caputo derivative for a single point `x_point`.
        This is analogous to `_fracdiff`.
        """
        # The Caputo derivative at t=0 is 0 by definition.
        if x_point == 0:
            return torch.tensor(0.0, dtype=x_point.dtype, device=x_point.device)

        # 1. Transform the Gauss-Jacobi nodes from [-1, 1] to the integration domain [0, x_point].
        t = 0.5 * x_point * (self.u + 1)

        # 2. To compute the derivative of `func` with respect to its input `t` using
        #    autograd, `t` must be a leaf tensor that requires a gradient.
        t_grad = t.clone().detach().requires_grad_(True).reshape(-1, 1)

        # 3. Evaluate the function at the quadrature points.
        func_vals_at_t = func(t_grad)

        # 4. Compute the first derivative of the function `func` at the quadrature points `t`.
        #    This uses the `diff` method provided by the BaseIE class.
        f_prime_vals = self.diff(func_vals_at_t, t_grad)

        # 5. Calculate the integral sum using the quadrature weights.
        integral_sum = torch.sum(self.w * f_prime_vals.flatten())

        # 6. Apply the scaling factor from the change of variables.
        result = (x_point / 2) ** (1 - self.alpha) * integral_sum

        # 7. Apply the final Gamma function scaling.
        return (1.0 / gamma(1 - self.alpha)) * result
