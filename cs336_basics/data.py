import os
import typing

import numpy as np
import torch
from numpy import typing as npt
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device | str
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

    # stack the list of NDArray in new axis, then passed to from_numpy
    x = torch.from_numpy(np.stack(x_list).astype(np.int64))
    y = torch.from_numpy(np.stack(y_list).astype(np.int64))
    # to gpu
    x_gpu = x.pin_memory().to(device, non_blocking=True)
    y_gpu = y.pin_memory().to(device, non_blocking=True)

    return (x_gpu, y_gpu)


class TextDataset(IterableDataset):
    def __init__(self, data: np.ndarray, context_length: int):
        self.dataset = data
        self.context_length = context_length
        # max sample numbers
        self.length = self.dataset.shape[0] - context_length - 1

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed % (2**32) if worker_info else None
        rng = np.random.default_rng(seed)

        while True:
            idx = rng.integers(low=0, high=self.length)

            x = self.dataset[idx : idx + self.context_length]
            y = self.dataset[idx + 1 : idx + self.context_length + 1]

            yield (
                torch.from_numpy(x.astype(np.int64)),
                torch.from_numpy(y.astype(np.int64))
            )


def get_batch_iterator(data, batch_size, context_length, device, num_workers=8):
    tds = TextDataset(data, context_length)
    dataloader = DataLoader(
        tds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0 
    )

    for x, y in dataloader:
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    checkpont = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpont, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
):
    checkpont = torch.load(src)
    model.load_state_dict(checkpont["model"])
    if optimizer:
        optimizer.load_state_dict(checkpont["optimizer"])
    return checkpont["iteration"]
