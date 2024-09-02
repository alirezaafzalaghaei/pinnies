import torch
from torch import optim
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseIE(ABC):
    def diff(self, y: torch.Tensor, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        """
        Compute the nth-order derivative of y with respect to x.

        Args:
            y (torch.Tensor): The dependent variable.
            x (torch.Tensor): The independent variable.
            n (int, optional): The order of the derivative. Defaults to 1.

        Returns:
            torch.Tensor: The computed derivative.
        """
        z = y
        for _ in range(n):
            z = torch.autograd.grad(
                z, x, grad_outputs=torch.ones_like(y), create_graph=True
            )[0]
        return z

    def get_loss(self, scale: float = 1e6) -> torch.Tensor:
        """
        Calculate the loss based on the residuals.

        Args:
            scale (float, optional): Scaling factor for the loss. Defaults to 1e6.

        Returns:
            torch.Tensor: The computed loss.
        """
        res = torch.cat(self.residual())
        return scale * (res**2).mean()

    def closure(self) -> torch.Tensor:
        """
        Compute the closure for the optimizer, typically used in LBFGS optimization.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.get_loss()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        return loss

    def solve(
        self, epochs: int, learning_rate: float = 0.1, verbose: bool = True
    ) -> float:
        """
        Train the model using a combination of LBFGS and Adam optimizers.

        Args:
            epochs (int): The number of training epochs.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.1.
            verbose (bool, optional): Whether to display a progress bar. Defaults to True.

        Returns:
            float: The final validation loss if exact solution is available, else None.
        """
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")

        self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.model.parameters(), lr=learning_rate)

        pbar = tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)

        for i in pbar:
            loss = self.get_loss()
            pbar.set_description(
                f"Validation: %.2e" % self.validation()
                if hasattr(self, "exact")
                else f"Loss: %.2e" % loss
            )

            self.optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer2.step()

        pbar = tqdm(range(1, epochs + 1)) if verbose else range(1, epochs + 1)

        for i in pbar:
            self.optimizer.step(self.closure)
            if verbose:
                pbar.set_description(
                    f"Validation: %.2e" % self.validation()
                    if hasattr(self, "exact")
                    else f"Loss: %.2e" % self.get_loss()
                )

        return self.validation() if hasattr(self, "exact") else None

    def predict(self, x):
        return self.model.forward(x)

    def generate_test(self, N=10):
        d = len(self.domain)
        points = torch.empty((N, d))

        for i in range(d):
            a, b = self.domain[i]
            points[:, i] = a + (b - a) * torch.rand(N)

        return points

    def validation(self, x_test=None):
        if x_test is None:
            x_test = self.generate_test()

        exact = self.exact(x_test).flatten()
        predict = self.predict(x_test).detach().numpy().flatten()
        error = exact - predict
        MAE = torch.abs(error).mean()
        return MAE

    def _shift_and_scale(
        self, roots: np.ndarray, lower: float, upper: float
    ) -> torch.Tensor:
        """
        Shift and scale roots to the domain [lower, upper].

        Args:
            roots (np.ndarray): The Legendre roots.
            lower (float): The lower bound of the domain.
            upper (float): The upper bound of the domain.

        Returns:
            torch.Tensor: Scaled roots.
        """
        return torch.tensor(
            (upper - lower) / 2 * roots + (lower + upper) / 2,
            requires_grad=True,
            dtype=torch.float32,
        )

    @abstractmethod
    def residual(self) -> List[torch.Tensor]:
        """
        Abstract method to compute the residuals.

        Returns:
            List[torch.Tensor]: A list of residuals.
        """
        pass
