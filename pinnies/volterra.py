from .ie import IE1D, IE2D, IE3D
import torch
from scipy.special import roots_legendre
from typing import Tuple


class Volterra1D(IE1D):
    def __init__(self, Omega: Tuple[float, float], N: int):
        """
        Initialize the Volterra1D class.

        Args:
            Omega (Tuple[float, float]): The domain of the function.
            N (int): Number of discretization points.
        """
        super().__init__(Omega, N)
        self.x, self.X, self.T, self.weights = self.compute_matrices()

    def compute_matrices(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the matrices required for Volterra1D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Points, grids, transformed grids, and weights.
        """
        roots, weights = roots_legendre(self.N)
        a, b = self.domain[0]
        weights = torch.tensor(weights, dtype=torch.float).reshape(-1, 1)
        shifted_roots = (b - a) / 2 * roots + (a + b) / 2
        r = torch.tensor(shifted_roots, requires_grad=True).float().reshape(-1, 1)
        T_v = torch.zeros((self.N, self.N))

        for i in range(1, self.N + 1):
            roots = torch.tensor(roots_legendre(self.N)[0], dtype=torch.float)
            ai, bi = 0, r[i - 1]
            shifted_roots = (bi - ai) / 2 * roots + (ai + bi) / 2
            T_v[i - 1, :] = shifted_roots

        X_v, _ = torch.meshgrid(r.flatten(), r.flatten(), indexing="ij")
        return r, X_v, T_v, weights

    def predict_on_T(self) -> torch.Tensor:
        """
        Predict using the transformed grid T.

        Returns:
            torch.Tensor: Predictions reshaped to match the grid dimensions.
        """
        t = self.T.reshape(-1, 1)
        return self.predict(t).reshape(*self.T.shape)

    def quad(self, integrand: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Perform quadrature on the given integrand.

        Args:
            integrand (torch.Tensor): The integrand to integrate.
            a (float): The start of the integration interval.
            b (float): The end of the integration interval.

        Returns:
            torch.Tensor: The result of the quadrature.
        """
        return (b - a) / 2 * integrand @ self.weights


class Volterra2D(IE2D):
    def __init__(self, domain: Tuple[Tuple[float, float], Tuple[float, float]], N: int):
        """
        Initialize the Volterra2D class.

        Args:
            domain (Tuple[Tuple[float, float], Tuple[float, float]]): The 2D domain of the function.
            N (int): Number of discretization points in each dimension.
        """
        super().__init__(domain, N)
        (
            self.x,
            self.X,
            self.Y,
            self.S,
            self.T,
            self.ST,
            self.bdx,
            self.bdy,
            self.weights,
        ) = self.compute_matrices()

    def compute_matrices(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute the matrices required for Volterra2D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Points, grids, transformed grids, weights, etc.
        """
        roots, weights = roots_legendre(self.N)
        a, b = self.domain[0]
        c, d = self.domain[1]
        weights = torch.tensor(weights, dtype=torch.float)
        x = self._shift_and_scale(roots, a, b)
        y = self._shift_and_scale(roots, c, d)

        XX, YY = torch.meshgrid(x, y, indexing="ij")
        X = XX.reshape(-1, 1)
        Y = YY.reshape(-1, 1)
        r = torch.cat((X, Y), 1)

        R1, R2 = self._compute_rotated_grids(x, y)
        RR1, RR2 = torch.meshgrid(R1.flatten(), R2.flatten(), indexing="ij")
        r1_r2 = torch.cat((RR1.reshape(-1, 1), RR2.reshape(-1, 1)), 1)

        X, Y, S, T = torch.meshgrid(
            x.flatten(), y.flatten(), x.flatten(), y.flatten(), indexing="ij"
        )
        S, T = R1, R2
        X = X.permute(3, 2, 0, 1)
        return r, X, Y, S, T, r1_r2, x.reshape(1, -1), y.reshape(1, 1, -1), weights

    def predict_on_ST(self) -> torch.Tensor:
        """
        Predict using the transformed grid ST.

        Returns:
            torch.Tensor: Predictions reshaped and permuted to match the grid dimensions.
        """
        return (
            self.predict(self.ST)
            .reshape(self.N, self.N, self.N, self.N)
            .permute(3, 2, 0, 1)
        )

    def _compute_rotated_grids(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotated grids R1 and R2 for Volterra2D.

        Args:
            x (torch.Tensor): The x points.
            y (torch.Tensor): The y points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated grids R1 and R2.
        """
        R1 = torch.zeros((self.N, self.N))
        R2 = torch.zeros((self.N, self.N))

        for i in range(1, self.N + 1):
            roots = torch.tensor(roots_legendre(self.N)[0], dtype=torch.float)
            R1[i - 1, :] = self._shift_and_scale(roots.numpy(), 0, x[i - 1].item())
            R2[i - 1, :] = self._shift_and_scale(roots.numpy(), 0, y[i - 1].item())

        R1 = R1.float().reshape(self.N, self.N, 1, 1).permute(3, 2, 0, 1)
        R2 = R2.float().reshape(self.N, self.N, 1, 1)

        return R1, R2

    def quad(self, integrand: torch.Tensor, a: float, b: float) -> torch.Tensor:
        """
        Perform quadrature on the given integrand.

        Args:
            integrand (torch.Tensor): The integrand to integrate.
            a (float): The start of the integration interval.
            b (float): The end of the integration interval.

        Returns:
            torch.Tensor: The result of the quadrature.
        """
        return (b - a) / 2 * (integrand @ self.weights)
