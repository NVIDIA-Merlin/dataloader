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
import warnings
from functools import partial

from merlin.core.compat.tensorflow import tensorflow as tf
from merlin.dataloader.loader_base import LoaderBase
from merlin.table import TensorColumn, TensorflowColumn, TensorTable
from merlin.table.conversions import _dispatch_dlpack_fns, convert_col


class Loader(LoaderBase, tf.keras.utils.Sequence):
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
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            seed_fn,
            parts_per_chunk,
            global_size,
            global_rank,
            drop_last,
            transforms,
            device,
        )

        self._validate_batch_size(batch_size)

        self.create_table = partial(TensorTable, _unsafe=True)
        self.create_column = partial(TensorColumn, _unsafe=True)
        column = self.create_column(self.array_lib().array([]))
        _to_dlpack_fn, _from_dlpack_fn = _dispatch_dlpack_fns(column, TensorflowColumn)
        self.convert_col = partial(
            convert_col, _to_dlpack_fn=_to_dlpack_fn, _from_dlpack_fn=_from_dlpack_fn, _unsafe=True
        )

    def _validate_batch_size(self, batch_size):
        is_power_of_two = batch_size & (batch_size - 1) == 0
        if self.device != "cpu" and (batch_size < 16 or not is_power_of_two):
            warnings.warn(
                "Due to a CUDA memory alignment issue in some Tensorflow "
                "operations such as Embedding ops, we recommend that "
                "'batch_size' be at least 16 and also a power of two. "
                "Please change 'batch_size' to a number that is a power of "
                "two that is greater than or equal to 16.",
                UserWarning,
            )

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
        return self.__next__()

    def __next__(self):
        """Get the next batch from the dataloader"""
        converted_batch = self.convert_batch(super().__next__())
        for map_fn in self._map_fns:
            converted_batch = map_fn(*converted_batch)

        return converted_batch

    def peek(self):
        """Grab the next batch from the dataloader
        without removing it from the queue"""
        converted_batch = self.convert_batch(self._peek_next_batch())
        for map_fn in self._map_fns:
            converted_batch = map_fn(*converted_batch)

        return converted_batch

    def on_epoch_end(self):
        self.stop()

    def convert_batch(self, batch):
        """Returns a batch after it has been converted to the appropriate tensor
        column type and then formats it in a flat dictionary which makes list
        columns into values and offsets as separate entries.

        Parameters
        ----------
        batch : tuple
            Tuple of dictionary inputs and n-dimensional array of targets

        Returns
        -------
        Tuple
            A tuple of dictionary inputs, with lists split as values and offsets,
            and targets as an array
        """
        inputs, targets = batch
        column_type = TensorflowColumn

        tf_inputs = {}
        if inputs is not None:
            inputs_table = self.create_table(inputs)
            for col_name, col in inputs_table.items():
                tf_inputs[col_name] = self.convert_col(col, column_type)

        tf_target = None
        if targets is not None:
            if isinstance(targets, dict):
                targets_table = self.create_table(targets)
                tf_targets = {}
                for col_name, col in targets_table.items():
                    tf_targets[col_name] = self.convert_col(col, column_type)
                tf_target = self.create_table(tf_targets).to_dict()
            else:
                targets_col = self.create_column(targets)
                tf_target = self.convert_col(targets_col, column_type).values

        return (self.create_table(tf_inputs).to_dict(), tf_target)

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self


class KerasSequenceValidater(tf.keras.callbacks.Callback):
    # TODO: document
    _supports_tf_logs = True

    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_epoch_end(self, epoch, logs=None):
        """Callback that runs at the end of an epoch.

        Parameters
        ----------
        epoch : int
            Integer representing the current epoch.
        logs : Dict, optional
            dictionary of logs collected, to be added to, by default None

        Returns
        -------
        logs
            Dictionary with results from callback.
        """
        logs = logs if logs is not None else {}
        for X, y_true in self.dataloader:
            y_pred = self.model(X)

            # TODO: how do we want to handle the multi-output case?
            for metric in self.model.metrics:
                metric.update_state(y_true, y_pred)

        set_logs = {}
        for metric in self.model.metrics:
            set_logs[f"val_{metric.name}"] = metric.result().numpy()
        logs.update(set_logs)
        print(set_logs)
        return logs
