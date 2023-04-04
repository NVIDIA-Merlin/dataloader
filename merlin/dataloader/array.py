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
from merlin.core.dispatch import annotate
from merlin.dataloader.loader_base import LoaderBase


class ArrayLoaderBase(LoaderBase):
    """Base class containing common functionality between the PyTorch and TensorFlow dataloaders."""

    def _peek_next_batch(self):
        """Return next batch without advancing the iterator."""
        if self._workers is None:
            ArrayLoaderBase.__iter__(self)

        # get the first chunks
        if self._batch_itr is None:
            self._fetch_chunk()

        # try to iterate through existing batches
        try:
            batch = next(self._batch_itr)
            self._batch_itr = itertools.chain([batch], self._batch_itr)

        except StopIteration:
            # anticipate any more chunks getting created
            # if not, raise the StopIteration
            if not self._working and self._buff.empty:
                self._workers = None
                self._batch_itr = None
                raise

            # otherwise get the next chunks and return
            # the first batch
            self._fetch_chunk()
            batch = next(self._batch_itr)
            self._batch_itr = itertools.chain([batch], self._batch_itr)

        return batch

    def _get_next_batch(self):
        """
        adding this cheap shim so that we can call this
        step without it getting overridden by the
        framework-specific parent class's `__next__` method.
        TODO: can this be better solved with a metaclass
        implementation? My gut is that we don't actually
        necessarily *want*, in general, to be overriding
        __next__ and __iter__ methods
        """
        # we've never initialized, do that now
        # need this because tf.keras.Model.fit will
        # call next() cold
        if self._workers is None:
            ArrayLoaderBase.__iter__(self)

        # get the first chunks
        if self._batch_itr is None:
            self._fetch_chunk()

        # try to iterate through existing batches
        try:
            batch = next(self._batch_itr)
        except StopIteration:
            # anticipate any more chunks getting created
            # if not, raise the StopIteration
            if not self._working and self._buff.empty:
                self._workers = None
                self._batch_itr = None
                raise

            # otherwise get the next chunks and return
            # the first batch
            self._fetch_chunk()
            batch = next(self._batch_itr)
        # if batch[0] is empty but other exist
        for sub in batch:
            if sub is not None and len(sub) > 0:
                self.num_rows_processed += len(sub)
                break
        return batch

    def _split_values(self, tensor, values_per_batch, axis=0):
        # splits are like offsets but without the first and last entry
        splits = list(itertools.accumulate(values_per_batch))[:-1]
        return self.array_lib().split(tensor, splits, axis=axis)

    def _subtract_offsets(self, offsets_grouped_by_batch):
        subtracted_offsets_grouped_by_batch = []
        for idx, batch_offsets in enumerate(offsets_grouped_by_batch):
            if idx != 0:
                previous_batch_offsets = offsets_grouped_by_batch[idx - 1]
                batch_offsets = batch_offsets - previous_batch_offsets[-1]
            subtracted_offsets_grouped_by_batch.append(batch_offsets)
        return subtracted_offsets_grouped_by_batch

    @annotate("make_tensors", color="darkgreen", domain="merlin_dataloader")
    def make_tensors(self, gdf, use_row_lengths=False):
        """Yields batches of tensors from a dataframe

        Parameters
        ----------
        gdf : DataFrame
            A dataframe type object.
        use_row_lengths : bool, optional
            Enable using row lengths instead of offsets for list columns, by default False

        Returns
        -------
        Dict[Tensors]
            A dictionary of the column tensor representations.

        """
        tensors_by_name = self._convert_df_to_tensors(gdf)
        rows_per_batch = self._get_rows_per_batch(len(gdf))

        tensor_batches = {}

        for tensor_key, tensor_value in tensors_by_name.items():
            if isinstance(tensor_value, tuple):
                # List feature
                full_tensor_values, full_tensor_offsets = tensor_value

                splits = list(itertools.accumulate(rows_per_batch))

                offsets_grouped_by_batch = []
                if splits:
                    for idx, split in enumerate([0] + splits[:-1]):
                        start = split
                        end = splits[idx] + 1
                        offsets_grouped_by_batch.append(full_tensor_offsets[start:end])

                subtracted_offsets_grouped_by_batch = self._subtract_offsets(
                    offsets_grouped_by_batch
                )
                num_values_per_batch = [
                    int(batch_offsets[-1]) for batch_offsets in subtracted_offsets_grouped_by_batch
                ]

                batch_values = self._split_values(full_tensor_values, num_values_per_batch)
                tensor_batches[tensor_key] = {
                    "values": batch_values,
                    "offsets": subtracted_offsets_grouped_by_batch,
                }
            else:
                # Scalar feature
                num_values_per_batch = rows_per_batch
                tensor_batches[tensor_key] = self._split_values(tensor_value, num_values_per_batch)

        for batch_idx in range(len(rows_per_batch)):
            batch = {}
            for tensor_key in tensors_by_name:
                tensor_value = tensor_batches[tensor_key]
                if isinstance(tensor_value, dict):
                    full_tensor_values = tensor_value["values"][batch_idx]
                    offsets = tensor_value["offsets"][batch_idx]
                    batch[tensor_key] = full_tensor_values, offsets
                else:
                    batch[tensor_key] = tensor_value[batch_idx]

            yield self._process_batch(batch)


class ArrayLoader(ArrayLoaderBase):
    """
    NumPy/CuPy Array dataloader

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
        """Number of batches in the dataloader."""
        return ArrayLoaderBase.__len__(self)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Note: This returns the next batch in the iterator.
              Not the batch at position `index`.
              This is because the dataloader is implemented as an iterator and
              don't currently support fetching a batch by index.
        """
        return ArrayLoaderBase.__next__(self)

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

        # if you have one series in a dataframe pull that series out
        # otherwise you will add a dimension to the column [[1,2,3]]
        try:
            if len(df_or_series.columns) == 1:
                df_or_series = df_or_series.iloc[:, 0]
        except AttributeError:
            pass

        if self.device == "cpu":
            tensor = df_or_series.to_numpy()
        else:
            tensor = df_or_series.to_cupy()

        return tensor

    def _cast_to_numpy_dtype(self, dtype):
        return dtype

    def _to_sparse_tensor(self, values_offset, column_name):
        raise NotImplementedError("Sparse support isn't implemented yet for the array dataloader")

    def _reshape_dim(self, tensor):
        return self.array_lib().reshape(tensor, [-1])

    def _row_lengths_to_offsets(self, row_lengths):
        zero_value = self.array_lib().array([0], dtype=row_lengths.dtype)
        if len(row_lengths.shape) == 2:
            zero_value = self.array_lib().expand_dims(zero_value, axis=0)
        return self.array_lib().concatenate(
            [zero_value, self.array_lib().cumsum(row_lengths)], axis=0
        )
