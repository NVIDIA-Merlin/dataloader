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
from merlin.core.compat import tensorflow as tf
from merlin.dataloader.array import Loader as ArrayLoader
from merlin.table import Device, NumpyColumn, TensorColumn, TensorflowColumn, TensorTable
from merlin.table.conversions import convert_col


class TFArrayDataloader(ArrayLoader, tf.keras.utils.Sequence):
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
        return self.convert_batch(self._peek_next_batch())

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
        target_column_type = TensorflowColumn

        tf_inputs = {}
        if inputs is not None:
            inputs_table = TensorTable(inputs, _unsafe=True)
            target_column_type = (
                TensorflowColumn if Device.GPU == inputs_table.device else NumpyColumn
            )
            for col_name, col in inputs_table.items():
                tf_inputs[col_name] = convert_col(col, target_column_type)

        tf_target = None
        if targets is not None:
            if isinstance(targets, dict):
                targets_table = TensorTable(targets, _unsafe=True)
                tf_targets = {}
                for col_name, col in targets_table.items():
                    tf_targets[col_name] = convert_col(col, target_column_type)
                tf_target = TensorTable(tf_targets).to_dict()
            else:
                targets_col = TensorColumn(targets, _unsafe=True)
                tf_target = convert_col(targets_col, target_column_type).values

        return (TensorTable(tf_inputs, _unsafe=True).to_dict(), tf_target)

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self
