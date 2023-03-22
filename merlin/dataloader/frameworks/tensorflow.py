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
from merlin.table import TensorColumn, TensorflowColumn, TensorTable
from merlin.table.conversions import convert_col


class TFArrayDataloader(ArrayLoader, tf.keras.utils.Sequence):
    def __next__(self):
        """Get the next batch from the dataloader"""
        return self.convert_batch(super().__next__())

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

        tf_inputs = {}
        if inputs:
            inputs_table = TensorTable(inputs)
            for col_name, col in inputs_table.items():
                tf_inputs[col_name] = convert_col(col, TensorflowColumn)

        tf_target = None
        if targets:
            targets_col = TensorColumn(targets)
            tf_target = convert_col(targets_col, TensorflowColumn).values

        return (TensorTable(tf_inputs).to_dict(), tf_target)
