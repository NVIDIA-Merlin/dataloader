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
import itertools

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

    def __len__(self):
        """Number of batches in the Sequence.

        Note: This also resets the loader state.
              Required because of the calls to `__getitem__`
              from keras prior to the start of the main loop
              through the loader.
        """
        LoaderBase.stop(self)
        return LoaderBase.__len__(self)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Note: This returns the next batch in the iterator.
              Not the batch at position `index`.
              This is because the dataloader is implemented as an iterator and
              don't currently support fetching a batch by index.
        """
        return LoaderBase.__next__(self)

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        yield dev

    def _split_fn(self, tensor, idx, axis=0):
        splits = list(itertools.accumulate(idx))[:-1]
        return self.array_lib().split(tensor, splits, axis=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        return self.array_lib().split(tensor, idx, axis=axis)

    def _to_tensor(self, df_or_series):
        if df_or_series.empty:
            return

        if self.device == "cpu":
            tensor = df_or_series.to_numpy()
        else:
            tensor = df_or_series.to_cupy()

        return tensor

    def _cast_to_numpy_dtype(self, dtype):
        return dtype

    def _to_sparse_tensor(self, values_offset, column_name):
        raise NotImplementedError("Sparse support isn't implemented yet for the array dataloader")
