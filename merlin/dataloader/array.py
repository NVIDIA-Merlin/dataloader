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

from merlin.core.compat import cupy as cp
from merlin.core.compat import numpy as np
from merlin.dataloader.loader_base import LoaderBase


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

    def array_lib(self):
        if self.device != "cpu":
            return cp
        else:
            return np

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        yield dev

    def _split_fn(self, tensor, idx, axis=0):
        return self.array_lib().split(tensor, idx, axis=axis)

    _tensor_split = _split_fn

    def _to_tensor(self, gdf):
        if gdf.empty:
            return

        if len(gdf.shape) == 1 or gdf.shape[1] == 1:
            dlpack = self._pack(gdf)
        elif gdf.shape[0] == 1:
            dlpack = self._pack(gdf.values[0])
        else:
            dlpack = self._pack(gdf.values.T)

        x = self._unpack(dlpack)

        if gdf.shape[0] == 1 and len(x.shape) != 2:
            # batch size 1 so got squashed to a vector
            x = x.reshape((1, x.shape[0]))
        elif len(gdf.shape) == 1 or len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        elif gdf.shape[1] > 1:
            x = x.T

        return x

    def _unpack(self, gdf):
        if self.device != "cpu":
            return self.array_lib().from_dlpack(gdf)
        else:
            # already numpy
            return gdf

    def _cast_to_numpy_dtype(self, dtype):
        return dtype

    def _to_sparse_tensor(self, values_offset, column_name):
        raise NotImplementedError("Sparse support isn't implemented yet for the array dataloader")
