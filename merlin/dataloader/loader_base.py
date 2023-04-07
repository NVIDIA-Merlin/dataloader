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
import copy
import itertools
import math
import queue
import threading
import warnings
from typing import List, Optional

from merlin.core.compat import HAS_GPU, cudf, cupy
from merlin.core.compat import numpy as np
from merlin.core.compat import pandas as pd
from merlin.core.dispatch import (
    annotate,
    concat,
    generate_local_seed,
    is_list_dtype,
    make_df,
    pull_apart_list,
)
from merlin.dag import BaseOperator, ColumnSelector, Graph, Node, ungroup_values_offsets
from merlin.dag.executors import LocalExecutor
from merlin.io import shuffle_df
from merlin.schema import Schema, Tags
from merlin.table import TensorTable


def _num_steps(num_samples, step_size):
    return math.ceil(num_samples / step_size)


class LoaderBase:
    """Base class containing common functionality between the PyTorch and TensorFlow dataloaders."""

    _use_row_lengths = False

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed_fn=None,
        parts_per_chunk=1,
        global_size=None,
        global_rank=None,
        drop_last=False,
        transforms=None,
        device=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed_fn = seed_fn
        self.parts_per_chunk = parts_per_chunk
        self.global_size = global_size or 1
        self.global_rank = global_rank or 0
        self.drop_last = drop_last
        self._map_fns = []

        device = device or 0
        self.device = "cpu" if not HAS_GPU or dataset.cpu else device

        if self.device == "cpu":
            self._array_lib = np
        else:
            self._array_lib = cupy

        self.indices = self._array_lib.arange(self.dataset.npartitions)

        if not dataset.schema:
            warnings.warn(
                "no schema associated with the input dataset. "
                "Calling dataset.infer_schema to automatically generate"
            )
            dataset.schema = dataset.infer_schema()

        self._epochs = 1
        self.num_rows_processed = 0

        self.__buff = None
        self.__buff_len = None
        self._batch_itr = None
        self._workers = None

        self._transforms = None
        self.executor = None
        self._transform_graph = None

        if transforms is not None:
            if isinstance(transforms, List):
                carry_node = Node(ColumnSelector("*"))
                for transform in transforms:
                    if not isinstance(transform, BaseOperator):
                        raise TypeError(
                            f"Detected invalid transform, {type(transform)},"
                            "we only support operators based on the merlin core"
                            "`BaseOperator`"
                        )
                    carry_node = carry_node >> transform
                transform_graph = Graph(carry_node)
            elif isinstance(transforms, Graph):
                transform_graph = transforms
            elif isinstance(transforms, Node):
                transform_graph = Graph(transforms)
            self._transform_graph = transform_graph
            self.executor = LocalExecutor()

        schema = dataset.schema
        self.input_schema = schema

    @property
    def transforms(self) -> Optional[Node]:
        return self._transforms

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    @property
    def _buff(self):
        if self.__buff is None:
            # we set size of chunk queue to 1 we only want one chunk in queue at a time.
            self.__buff = ChunkQueue(
                self,
                1,
                num_parts=self.parts_per_chunk,
                shuffle=self.shuffle,
                epochs=self._epochs,
            )
        return self.__buff

    @property
    def _buff_len(self):
        if self.__buff_len is None:
            # run once instead of every time len called
            self.__buff_len = len(self._buff)
        return self.__buff_len

    def epochs(self, epochs=1):
        """Create a dataloader that will efficiently run for more than one epoch.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs the dataloader should process data, by default 1

        Returns
        -------
        DataLoader
            return a dataloader that will run for user defined epochs.
        """
        if epochs == self._epochs:
            return self
        new_dataloader = copy.copy(self)
        new_dataloader._set_epochs(epochs)
        return new_dataloader

    def _set_epochs(self, epochs):
        self.stop()
        self.__buff = None
        self.__buff_len = None
        self._epochs = epochs

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Note: This returns the next batch in the iterator.
              Not the batch at position `index`.
              This is because the dataloader is implemented as an iterator and
              don't currently support fetching a batch by index.
        """
        return self.__next__()

    def __len__(self):
        batches = _num_steps(self._buff_len, self.batch_size)
        if self.drop_last and self._buff_len % self.batch_size > 0:
            batches = batches - 1
        return batches

    @property
    def _working(self):
        if self._workers is not None:
            return any(t.is_alive() for t in self._workers)
        return False

    def stop(self):
        """Halts and resets the initialization parameters of the dataloader."""
        # TODO: raise warning or even error if condition
        # isn't met?
        if self._workers is not None:
            if not self._buff.stopped:
                self._buff.stop()
            for t in self._workers:
                t.join()
            # remove joined threads from list
            self._workers = None
            self._buff.q_out.queue.clear()
        self._batch_itr = None

    def _indices_for_process(self):
        # this should be self.indices divided by total processes, global set
        if len(self.indices) < self.global_size:
            warnings.warn(
                f"""You have more processes({self.global_size}) than dataset
                    partitions({len(self.indices)}), reduce the number of processes."""
            )
            raise IndexError
        per_worker = _num_steps(len(self.indices), self.global_size)
        # identify process rank out of all processes (not local rank)
        start = self.global_rank * per_worker
        return self.indices[start : start + per_worker].tolist()

    @annotate("_shuffle_indices", color="darkgreen", domain="merlin_dataloader")
    def _shuffle_indices(self):
        generate_local_seed(self.global_rank, self.global_size)
        if self.seed_fn:
            new_seed = self.seed_fn()
            self._array_lib.random.seed(new_seed)
        self._array_lib.random.shuffle(self.indices)
        generate_local_seed(self.global_rank, self.global_size)

    def __iter__(self):
        self.stop()
        self.num_rows_processed = 0
        if self._buff.stopped:
            self._buff.start()

        # shuffle partition indices to bring disparate
        # parts of the dataset "close" to one another
        if self.shuffle:
            self._shuffle_indices()

        # build and start new threads for loading and
        # concatenating data
        self._workers = []
        t = threading.Thread(target=self._buff.load_chunks, args=(self.device,))
        t.daemon = True
        t.start()
        self._workers.append(t)
        return self

    def __next__(self):
        """Get the next batch."""
        return self._get_next_batch()

    def peek(self):
        """Get the next batch without advancing the iterator."""
        return self._peek_next_batch()

    def _data_iter(self, epochs):
        indices = self._indices_for_process()
        return self.dataset.to_iter(
            indices=indices, epochs=epochs, columns=self.dataset.schema.column_names
        )

    def _fetch_chunk(self):
        chunks = self._buff.get()
        if isinstance(chunks, Exception):
            self.stop()
            raise chunks
        self._batch_itr = iter(chunks)

    def _peek_next_batch(self):
        """Return next batch without advancing the iterator."""
        if self._workers is None:
            LoaderBase.__iter__(self)

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
            LoaderBase.__iter__(self)

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

    def _get_rows_per_batch(self, num_samples):
        """
        Helper function to build indices to pass
        to <torch|tf>.split functions for breaking
        up into batches
        """
        num_full_batches = _num_steps(num_samples, self.batch_size) - 1
        idx = [self.batch_size for _ in range(num_full_batches)]
        idx.append(num_samples - num_full_batches * self.batch_size)
        return idx

    def _to_tensor(self, df_or_series):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a cudf DataFrame to a
        tensor in the appropriate library, with an optional
        dtype kwarg to do explicit casting if need be
        """
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

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        """
        One of the mandatory functions a child class needs
        to implement. Maps from a GPU index to a framework
        context object for placing tensors on specific GPUs
        """
        yield dev

    def _cast_to_numpy_dtype(self, dtype):
        """
        Get the numpy dtype from the framework dtype.
        """
        raise dtype

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

    def _separate_list_columns(self, gdf):
        lists, scalars = [], []
        for col in gdf.columns:
            if is_list_dtype(gdf[col]):
                lists.append(col)
            else:
                scalars.append(col)
        return scalars, lists

    def array_lib(self):
        return self._array_lib

    def _split_fn(self, tensor, idx, axis=0):
        splits = list(itertools.accumulate(idx))[:-1]
        return self.array_lib().split(tensor, splits, axis=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        return self.array_lib().split(tensor, idx, axis=axis)

    @annotate("_convert_df_to_tensors", color="darkgreen", domain="merlin_dataloader")
    def _convert_df_to_tensors(self, gdf):
        """Convert a dataframe into framework tensors.
        Returns dictionary of tensors by feature name.
        Where scalar features are grouped under the same key (tuple of column names)
        when they share the same dtype.
        """
        tensors_by_name = {}
        for column_names in self.dtype_reverse_map.values():
            gdf_i = gdf[column_names]
            gdf.drop(columns=column_names, inplace=True)

            scalars, lists = self._separate_list_columns(gdf_i)

            if scalars:
                # split out cols and change all scalars
                # should always return dict column_name: values, offsets (optional)
                for column_name in column_names:
                    tensors_by_name[column_name] = self._to_tensor(gdf_i[[column_name]])

            if lists:
                # split out lists
                for column_name in lists:
                    column = gdf_i.pop(column_name)

                    if cudf and isinstance(column, cudf.Series):
                        is_fixed_length = (
                            not self.input_schema[column_name].shape.is_ragged
                            and (column.list.len() == column.list.len()[0]).all()
                        )
                        if is_fixed_length:
                            values = column.list.leaves.values.reshape(
                                -1, *cupy.array(column[0]).shape
                            )
                            tensors_by_name[column_name] = values
                            continue
                    elif isinstance(column, pd.Series):
                        if not self.input_schema[column_name].shape.is_ragged:
                            values = np.array(list(_array_leaves(column.values))).reshape(
                                -1, *np.array(column[0]).shape
                            )
                            tensors_by_name[column_name] = values
                            continue

                    leaves, col_offsets = pull_apart_list(column, device=self.device)

                    if isinstance(leaves[0], list):
                        leaves, nest_offsets = pull_apart_list(leaves, device=self.device)
                        col_offsets = nest_offsets.iloc[col_offsets[:]]

                    tensors_by_name[column_name] = self._to_tensor(leaves), self._to_tensor(
                        col_offsets
                    )

        return tensors_by_name

    @annotate("_process_batch", color="darkgreen", domain="merlin_dataloader")
    def _process_batch(self, tensors):
        X = ungroup_values_offsets(tensors)

        # Return a tensor if we have only one label column, but return a
        # dictionary of tensors if there are multiple label columns, since
        # dictionary output is required in Merlin Models and Transformers4Rec.
        # If a user is using a vanilla Keras model with multiple labels,
        # they would need to provide matching column names in the output layer
        # of the Keras model.
        if len(self.label_names) == 0:
            labels = None
        elif len(self.label_names) == 1:
            labels = X.pop(self.label_names[0])
        else:
            labels = {}
            for label in self.label_names:
                labels[label] = X.pop(label)

        if self.transforms:
            X = self.executor.transform(TensorTable(X), [self.transforms])

        return X, labels

    @property
    def schema(self) -> Schema:
        """Get input schema of data to be loaded

        Returns
        -------
        ~merlin.schema.Schema
            Schema corresponding to the data
        """
        warnings.warn(
            "This `schema` property is deprecated and will be removed in a future version. "
            "Please use either the `input_schema` or `output_schema` property instead."
        )
        return self._input_schema

    @property
    def input_schema(self) -> Schema:
        """Get input schema of data to be loaded.

        If there are no transforms then this will be the same as the output schema.

        Returns
        -------
        ~merlin.schema.Schema
            Schema corresponding to the data that will be loaded  prior to any transforms.
        """
        return self._input_schema

    @property
    def output_schema(self) -> Schema:
        """Get output schema of data being loaded.

        When there are transforms defined that change the features being loaded,
        This output schema is intended to account for this and should match
        the features returned by the loader. If there are no transforms then this
        will be the same as the input schema.

        Returns
        -------
        ~merlin.schema.Schema
            Schema corresponding to the data that will be output by the loader
        """
        return self._output_schema

    @input_schema.setter
    def input_schema(self, value):
        """Set schema property
        Parameters
        ----------
        value : ~merlin.schema.Schema
            The schema corresponding to data to be loaded.
        Raises
        ------
        ValueError
            When value provided doesn't match expected type
        """
        if self._batch_itr is not None:
            raise RuntimeError(
                "Setting the input_schema after the dataloader has started is not supported. "
                "If you would like to change the input_schema "
                "please change before reading the first batch. "
            )
        if not isinstance(value, Schema):
            raise ValueError(
                "schema value on loader must be of type merlin.io.Schema. "
                f"provided: {type(value)}"
            )
        self._input_schema = value

        self.cat_names = (
            value.select_by_tag(Tags.CATEGORICAL).excluding_by_tag(Tags.TARGET).column_names
        )
        self.cont_names = (
            value.select_by_tag(Tags.CONTINUOUS).excluding_by_tag(Tags.TARGET).column_names
        )
        self.label_names = value.select_by_tag(Tags.TARGET).column_names

        self.dtype_reverse_map = {}

        for col_name, col_spec in self._input_schema.column_schemas.items():
            if col_spec.dtype not in self.dtype_reverse_map:
                self.dtype_reverse_map[col_spec.dtype] = [col_name]
            else:
                self.dtype_reverse_map[col_spec.dtype].append(col_name)

        if self._transform_graph is not None:
            self._transforms = self._transform_graph.construct_schema(
                self._input_schema
            ).output_node
            self._output_schema = self._transforms.output_schema
        else:
            self._output_schema = self._input_schema

        if len(list(self.dtype_reverse_map.keys())) == 0:
            raise ValueError(
                "Neither Categorical or Continuous columns were found by the dataloader. "
                "You must either specify the cat_names, cont_names and "
                "label_names properties or supply a schema.pbtxt file in dataset directory."
            )


class ChunkQueue:
    """This class takes partitions (parts) from an merlin.io.Dataset
    and concatenates them into a cudf dataframe "chunk." This chunk
    is subsequently transformed into its tensor representation using
    the iterator's transform.

    Parameters
    ----------
    qsize: int
        Maximum number of elements to hold in the buffer at one time.
    num_parts : int
        Number of partitions from the iterator, a merlin.io.Dataset to
        concatenate into a "chunk."
    shuffle : bool
        Enable or disable chunk-level shuffling.
    put_wait: float
        Specifies the timeout to wait for a full queue to open up before checking
        for errors and trying again.
    """

    def __init__(self, dataloader, qsize, num_parts=1, shuffle=False, put_wait=1e-6, epochs=1):
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.put_wait = put_wait
        self.q_out = queue.Queue(qsize)
        self._stop_event = threading.Event()
        self.itr = dataloader._data_iter(epochs)
        self.dataloader = dataloader

    def __len__(self):
        return len(self.itr)

    @property
    def stopped(self):
        return self._stop_event.is_set()

    @property
    def empty(self):
        return self.q_out.empty()

    def get(self):
        return self.q_out.get()

    def put(self, packet):
        while True:
            if self.stopped:
                return True

            try:
                self.q_out.put(packet, timeout=self.put_wait)
                return False
            except queue.Full:
                continue

    @annotate("batch", color="darkgreen", domain="merlin_dataloader")
    def batch(self, itr):
        """Iterates through gpu_mem_frac size chunks of dataset
        and concatenates every `num_parts` of them.
        """
        current = []
        while True:
            try:
                value = next(itr)
            except StopIteration:
                if len(current) > 0:
                    yield current
                break

            current.append(value)
            if len(current) == self.num_parts:
                yield current
                current = []

    @annotate("chunk_logic", color="darkgreen", domain="merlin_dataloader")
    def chunk_logic(self, itr):
        spill = None
        for chunks in self.batch(itr):
            if self.stopped:
                return

            if spill is not None and not spill.empty:
                chunks.insert(0, spill)

            chunks = concat(chunks)
            chunks.reset_index(drop=True, inplace=True)
            chunks, spill = self.get_batch_div_chunk(chunks, self.dataloader.batch_size)
            if self.shuffle:
                chunks = shuffle_df(chunks, keep_index=True)

            if len(chunks) > 0:
                chunks = self.dataloader.make_tensors(chunks, self.dataloader._use_row_lengths)
                # put returns True if buffer is stopped before
                # packet can be put in queue. Keeps us from
                # freezing on a put on a full queue
                if self.put(chunks):
                    return
            chunks = None
        # takes care final batch, which is less than batch size
        if not self.dataloader.drop_last and spill is not None and not spill.empty:
            spill = self.dataloader.make_tensors(spill, self.dataloader._use_row_lengths)
            self.put(spill)

    @annotate("load_chunks", color="darkgreen", domain="merlin_dataloader")
    def load_chunks(self, dev):
        try:
            itr = iter(self.itr)
            if self.dataloader.device != "cpu":
                with self.dataloader._get_device_ctx(dev):
                    self.chunk_logic(itr)
            else:
                self.chunk_logic(itr)
        except Exception as e:  # pylint: disable=broad-except
            self.put(e)

    # For when an iterator is stopped before iteration is complete.
    def stop(self):
        self._stop_event.set()
        # TODO: should we be clearing? I can imagine a world where
        # you want the thread to stop but still want to grab
        # data out of the buffer
        self.q_out.queue.clear()

    def start(self):
        self._stop_event.clear()

    def get_batch_div_chunk(self, chunks, batch_size):
        # TODO: is there a way to do this using cupy?
        spill_idx = int(chunks.shape[0] / batch_size) * batch_size
        spill = make_df(chunks.iloc[spill_idx:])
        chunks = make_df(chunks.iloc[:spill_idx])
        if not chunks.empty:
            chunks.reset_index(drop=True, inplace=True)
        if not spill.empty:
            spill.reset_index(drop=True, inplace=True)
        return chunks, spill


def _array_leaves(x):
    if isinstance(x, (list, np.ndarray)):
        for element in x:
            yield from _array_leaves(element)
    else:
        yield x
