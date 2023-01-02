from torch.utils.data import Dataset
from torch import Tensor
from typing import *


class OneSampleDataset(Dataset):
    def __init__(self, x: Tensor, y: Optional[Tensor] = None):
        self.x = x
        self.y_available = y is not None
        if self.y_available:
            assert x.size(0) == y.size(0), "X and y must have equal shape."
            self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        if self.y_available:
            return self.x[index], self.y[index]
        else:
            return self.x[index]


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        seq_len: int = 1,
        y_position: Optional[int] = None,
    ):
        """
        Creates a sequence of sliding windows from the original X array, and maps each
            window to a target on a specified position within the window.

        Parameters
        ----------
        x: Tensor with features of shape (n_samples, n_features)
        y: Tensor with targets of shape (n_samples, 1)
        seq_len: Window length
        y_position: Position within the window that corresponds to the target.
            By default, the target is mapped to the last window element.
            Can be anything between 0 and seq_len-1.
        """
        self.seq_len = seq_len
        self.x = self.apply_sliding_window_to_x(x)
        self.y_available = y is not None
        if self.y_available:
            assert x.size(0) == y.size(0), "X and y must have equal shape."
            self.y_position = y_position if y_position is not None else seq_len - 1
            assert (
                self.y_position < self.seq_len
            ), f"Y position must be between 0 and sequence length - 1."
            self.y = self.get_y_per_window(y)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        if self.y_available:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

    def apply_sliding_window_to_x(self, x):
        """Transforming an array into a sequence of sliding windows of size:
        (batch, seq_len, n_features)."""
        return x.unfold(dimension=0, size=self.seq_len, step=1).transpose(2, 1)

    def get_y_per_window(self, y):
        start = self.y_position
        end = -self.seq_len + start + 1 if start + 1 != self.seq_len else None
        return y[start:end]
