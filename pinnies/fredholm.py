from .ie import IE1D, IE2D, IE3D
import torch
from scipy.special import roots_legendre

import torch
from scipy.special import roots_legendre
from typing import Tuple


class Fredholm1D(IE1D):
    def __init__(self, Omega: Tuple[float, float], N: int):
        """
        Initialize the Fredholm1D class.

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
        Compute the matrices required for Fredholm1D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Returns the points, grid, transformed grid, and weights.
        """
        roots, weights = roots_legendre(self.N)
        a, b = self.domain[0]
        weights = torch.tensor(weights, dtype=torch.float).reshape(-1, 1)
        shifted_roots = (b - a) / 2 * roots + (a + b) / 2
        r = torch.tensor(shifted_roots, requires_grad=True).float().reshape(-1, 1)
        X_f, T_f = torch.meshgrid(r.flatten(), r.flatten(), indexing="ij")
        return r, X_f, T_f, weights

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
        return (b - a) / 2 * integrand.T @ self.weights


class Fredholm2D(IE2D):
    def __init__(self, domain: Tuple[Tuple[float, float], Tuple[float, float]], N: int):
        """
        Initialize the Fredholm2D class.

        Args:
            domain (Tuple[Tuple[float, float], Tuple[float, float]]): The 2D domain of the function.
            N (int): Number of discretization points in each dimension.
        """
        super().__init__(domain, N)
        self.x, self.X, self.Y, self.S, self.T, self.weights = self.compute_matrices()

    def compute_matrices(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute the matrices required for Fredholm2D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Points, grids, transformed grids, and weights.
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

        X, Y, S, T = torch.meshgrid(
            x.flatten(), y.flatten(), x.flatten(), y.flatten(), indexing="ij"
        )
        return r, X, Y, S, T, weights

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


class Fredholm3D(IE3D):
    def __init__(
        self,
        domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        N: int,
    ):
        """
        Initialize the Fredholm3D class.

        Args:
            domain (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
                The 3D domain of the function.
            N (int): Number of discretization points in each dimension.
        """
        super().__init__(domain, N)
        (
            self.x,
            self.X,
            self.Y,
            self.Z,
            self.R,
            self.S,
            self.T,
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
    ]:
        """
        Compute the matrices required for Fredholm3D.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Points, grids, transformed grids, and weights.
        """
        # Compute roots and weights for quadrature
        roots, weights = roots_legendre(self.N)

        # Compute 1D grids along each dimension
        x = self._shift_and_scale(roots, *self.domain[0])
        y = self._shift_and_scale(roots, *self.domain[1])
        z = self._shift_and_scale(roots, *self.domain[2])

        # Create 3D mesh grids
        XX, YY, ZZ = torch.meshgrid(x, y, z, indexing="ij")
        X = XX.reshape(-1, 1)
        Y = YY.reshape(-1, 1)
        Z = ZZ.reshape(-1, 1)

        # Combine the grids into a single tensor
        r = torch.cat((X, Y, Z), dim=1)

        # Compute full 6D grid for the kernel
        X, Y, Z, R, S, T = torch.meshgrid(
            x.flatten(),
            y.flatten(),
            z.flatten(),
            x.flatten(),
            y.flatten(),
            z.flatten(),
            indexing="ij",
        )

        # Convert weights to torch tensor
        weights = torch.tensor(weights, dtype=torch.float)

        return r, X, Y, Z, R, S, T, weights

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
