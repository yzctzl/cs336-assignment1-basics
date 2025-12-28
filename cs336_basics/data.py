import os
import typing

import numpy as np
import torch
from numpy import typing as npt
from torch import Tensor


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Tensor, Tensor]:
    """
        Given a dataset (a 1D numpy array of integers) and a desired batch size and
        context length, sample language modeling input sequences and their corresponding
        labels from the dataset.

        Args:
            dataset (np.array): 1D numpy array of integer token IDs in the dataset.
            batch_size (int): Desired batch size to sample.
            context_length (int): Desired context length of each sampled example.
            device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
                to place the sampled input sequences and labels on.

        Returns:
            Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
            is the sampled input sequences, and the second tuple item is the corresponding
            language modeling labels.
        
        When lazy-load(on-demand) huge dataset, via np.memmap or the flag mmap_mode='r' to np.load.
    """
    ds_lenth = dataset.shape[0]
    # any 1 <= i < n − m gives a valid training sequence, so sampling sequences are trivial
    idx = torch.randint(low=0, high=ds_lenth - context_length, size=(batch_size,))

    # slice numpy NDArray in place, it's cheap
    x_list = [dataset[i : i + context_length] for i in idx]
    y_list = [dataset[i + 1 : i + context_length + 1] for i in idx]

    # stack the list of NDArray in new axis, then passed to as_tensor
    x = torch.as_tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.as_tensor(np.stack(y_list), dtype=torch.long, device=device)

    return (x, y)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpont = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpont, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpont = torch.load(src)
    model.load_state_dict(checkpont["model"])
    optimizer.load_state_dict(checkpont["optimizer"])
    return checkpont["iteration"]
