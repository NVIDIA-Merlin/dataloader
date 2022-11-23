#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

from merlin.core.dispatch import HAS_GPU
from merlin.dataloader.loader_base import LoaderBase

numpy_to_torch_dtype_dict = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


class Loader(torch.utils.data.IterableDataset, LoaderBase):
    """This class creates batches of tensor. Each batch size is specified by the user.
    The data input requires a merlin.io.Dataset. Handles spillover to ensure all
    batches are the specified size until the final batch.

    Parameters
    ----------
    dataset: merlin.io.Dataset
        The dataset to load
    batch_size: int
        Number of rows to yield at each iteration
    shuffle: bool, default True
        Whether to shuffle chunks of batches before iterating through them.
    seed_fn: callable
        Function used to initialize random state
    parts_per_chunk: int
        Number of dataset partitions with size dictated by `buffer_size`
        to load and concatenate asynchronously. More partitions leads to
        better epoch-level randomness but can negatively impact throughput
    global_size: int, optional
        When doing distributed training, this indicates the number of total processes that are
        training the model.
    global_rank:
        When doing distributed training, this indicates the local rank for the current process.
    drop_last: bool, default False
        Whether or not to drop the last batch in an epoch. This is useful when you need to
        guarantee that each batch contains exactly `batch_size` rows - since the last batch
        will usually contain fewer rows.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        seed_fn=None,
        parts_per_chunk=1,
        global_size=None,
        global_rank=None,
        drop_last=False,
        transforms=None,
        device=None,
    ):
        LoaderBase.__init__(
            self,
            dataset,
            batch_size,
            shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            transforms=transforms,
            device=device,
        )

    def __iter__(self):
        return LoaderBase.__iter__(self)

    def _get_device_ctx(self, dev):
        if dev == "cpu":
            return torch.device("cpu")
        return torch.cuda.device(f"cuda:{dev}")

    def _unpack(self, dlpack):
        if self.device == "cpu":
            values = dlpack.values if hasattr(dlpack, "values") else dlpack
            dtype = values.dtype
            dtype = numpy_to_torch_dtype_dict[dtype.type] if hasattr(dtype, "type") else dtype
            if (
                len(values.shape) == 2
                and values.shape[1] == 1
                and isinstance(values[0], np.ndarray)
            ):
                return torch.squeeze(torch.Tensor(values)).type(dtype)
            return torch.Tensor(values).type(dtype)
        return from_dlpack(dlpack)

    def _to_tensor(self, gdf):
        return self._unpack(self._pack(gdf))

    def _split_fn(self, tensor, idx, axis=0):
        return torch.split(tensor, idx, dim=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        return torch.tensor_split(tensor, idx, axis=axis)

    def _pull_values_offsets(self, values_offset):
        # pull_values_offsets, return values offsets diff_offsets
        if isinstance(values_offset, tuple):
            values = values_offset[0].flatten()
            offsets = values_offset[1].flatten()
        else:
            values = values_offset.flatten()
            offsets = torch.arange(values.size()[0], device=self.device)
        num_rows = len(offsets)
        if HAS_GPU:
            offsets = torch.cat([offsets, torch.cuda.LongTensor([len(values)], device=self.device)])
        else:
            offsets = torch.cat([offsets, torch.LongTensor([len(values)])])
        diff_offsets = offsets[1:] - offsets[:-1]
        return values, offsets, diff_offsets, num_rows

    def _get_max_seq_len(self, diff_offsets):
        return int(diff_offsets.max())

    # Building the indices to reconstruct the sparse tensors

    def _get_indices(self, offsets, diff_offsets):
        row_ids = torch.arange(len(offsets) - 1, device=self.device)
        row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
        row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
        col_ids = torch.arange(len(row_offset_repeated), device=self.device) - row_offset_repeated
        indices = torch.cat([row_ids_repeated.unsqueeze(-1), col_ids.unsqueeze(-1)], axis=1)
        return indices

    def _build_sparse_tensor(
        self, values, offsets, diff_offsets, num_rows, seq_limit, sparse_as_dense
    ):
        indices = self._get_indices(offsets, diff_offsets)
        sparse_tensor = torch.sparse_coo_tensor(
            indices.T, values, torch.Size([num_rows, seq_limit]), device=self.device
        )
        if sparse_as_dense:
            sparse_tensor = sparse_tensor.to_dense()
        return sparse_tensor

    def _cast_to_numpy_dtype(self, dtype):
        """
        Get the numpy dtype from the framework dtype.
        """
        return torch_to_numpy_dtype_dict[dtype]


class DLDataLoader(torch.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.
    """

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)
