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
import contextlib

import jax.numpy as jnp
import numpy as np

from merlin.dataloader.loader_base import LoaderBase

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,redundant-keyword-arg


class Loader(LoaderBase):
    """
    Jax dataloader

    Parameters
    -------------
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

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        yield dev

    def _split_fn(self, tensor, idx, axis=0):
        if isinstance(idx, int):
            splits = jnp.split(tensor, idx, axis=axis)
        else:
            splits = jnp.split(tensor, np.cumsum(idx)[:-1], axis=axis)
        return splits

    _tensor_split = _split_fn

    def _sum(self, tensor):
        return tensor.sum()

    def _cast_to_numpy_dtype(self, dtype):
        # jax uses numpy dtypes, so this is kinda easy
        return dtype

    def _to_sparse_tensor(self, values_offset, column_name):
        raise NotImplementedError("Sparse support isn't implemented yet for the Jax dataloader")

    def _row_lengths_to_offsets(self, row_lengths):
        zero_value = jnp.array([0], dtype=row_lengths.dtype)
        if len(row_lengths.shape) == 2:
            zero_value = zero_value.reshape(-1, 1)
        return jnp.concatenate([zero_value, jnp.cumsum(row_lengths, axis=0)], axis=0)
