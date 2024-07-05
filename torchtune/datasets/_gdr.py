# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import torch
import torch
import numba
import numpy as np
from enum import Enum
from functools import partial
from torch.utils.data import Dataset

class DataLayout(Enum):
    token_bits = (0, 17)  # max number of tokens: 128k
    position_bits = (17, 30)  # max context length of 8k
    loss_target = (30, 31)  # just one bit.

def get_bits(data: torch.Tensor, layout: DataLayout):
    assert data.dtype == torch.long
    low, high = layout.value
    high_mask = (1 << high) - 1
    masked = torch.bitwise_and(data, high_mask)
    shifted = torch.bitwise_right_shift(masked, low)
    return shifted

def attn_mask_pos_(x, am):
    for i in range(len(x)):
        attn = False
        for j in range(0, len(x)):
            attn |= (i + 1) == j
            attn &= x[i] < x[j]
            am[j, i] = ~attn & (i != j)

attn_mask_pos_ = numba.jit(attn_mask_pos_)

def maybe_return_float_tensor(mask, return_float_tensor: bool, dtype):
    if return_float_tensor:
        return (mask * 1.0).masked_fill(mask, -1e9).to(dtype)
    return mask

def get_attention_mask_from_positions(positions: torch.Tensor, return_float_tensor: bool = False, dtype=torch.bfloat16):
    bz, seq_len = positions.shape
    positions = positions.numpy()
    res = np.empty(shape=(bz, seq_len, seq_len), dtype=bool)
    for i in range(len(positions)):
        attn_mask_pos_(positions[i], res[i])
    pt_mask = torch.tensor(res)
    return maybe_return_float_tensor(pt_mask, return_float_tensor, dtype)

def collate_batch(
    data: list[torch.LongTensor], ignore_idx: int = -100
) -> dict:
    data = torch.stack(list(data))
    tokens = get_bits(data, DataLayout.token_bits)
    input_ids = tokens[:, :-1]
    target_ids = tokens[:, 1:].clone()

    positions = get_bits(data, DataLayout.position_bits)[:,:-1]

    attn_mask = get_attention_mask_from_positions(positions).logical_not()

    loss_targets = get_bits(data[:, 1:], DataLayout.loss_target)

    target_ids[loss_targets == 0] = ignore_idx

    return {"tokens": input_ids, "labels": target_ids, "mask": attn_mask, "input_pos": positions}


class MemmappedDataset(Dataset):
    r"""
    Dataset that takes a memory mapped numpy array from disk and returns chunks of length context_size of it.
    """

    data: np.ndarray

    def __init__(self, path: str, n_tokens: int = 8192) -> None:
        super().__init__()
        self.path = path
        self.max_seq_len = (
            n_tokens  # naming needed for timestamp to work properly for tokens
        )
        # Note:
        # Torch will try to pickle this object, when workers > 0. (to send it to the worker threads)
        # If we memmap the entire dataset it will happily try to pickle it and dump the dataset to disk..
        # In order to work around this we set the data to zero.
        self.data = None

    def __getitem__(self, index: int):
        if self.data is None:
            # safe to store it now, we're on the worker thread.
            self.data = np.memmap(self.path, dtype=np.int32, mode="readonly")

        start = index * (self.max_seq_len + 1)
        end = (index + 1) * (self.max_seq_len + 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = torch.as_tensor(self.data[start:end], dtype=torch.long)
        return data

    def __len__(self):
        if self.data is None:
            # See above note, before we call the first __getitem__ we cannot store the mmap file.
            data = np.memmap(self.path, dtype=np.int32, mode="readonly")
            return data.shape[0] // (self.max_seq_len + 1)

        return self.data.shape[0] // (self.max_seq_len + 1)

class GroupedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_dataset_lengths = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, index: int):
        current_dataset_idx = int(np.where(index < self.cum_dataset_lengths)[0][0])
        offset = 0 if current_dataset_idx == 0 else int(self.cum_dataset_lengths[current_dataset_idx - 1])
        return self.datasets[current_dataset_idx][index - offset]

    def __len__(self) -> int:
        return int(self.cum_dataset_lengths[-1])

def gdr(
    path,
    max_seq_len: int = 8192,
) -> MemmappedDataset:

    return MemmappedDataset(
        n_tokens=max_seq_len,
        path=path,
    )

gdr = partial(gdr)
