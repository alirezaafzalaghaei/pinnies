import torch
from scipy.special import gamma


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
