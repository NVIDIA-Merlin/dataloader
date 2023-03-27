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
from merlin.core.compat import torch as th
from merlin.dataloader.array import Loader as ArrayLoader
from merlin.table import TensorColumn, TensorTable, TorchColumn
from merlin.table.conversions import _dispatch_dlpack_fns, convert_col


class TorchArrayDataloader(ArrayLoader, th.utils.data.IterableDataset):
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

        column = TensorColumn(self.array_lib().array([]))
        self._to_dlpack_fn, self._from_dlpack_fn = _dispatch_dlpack_fns(column, TorchColumn)

    def __next__(self):
        """Get the next batch from the dataloader"""
        converted_batch = self.convert_batch(
            super().__next__(), self._to_dlpack_fn, self._from_dlpack_fn
        )
        for map_fn in self._map_fns:
            converted_batch = map_fn(*converted_batch)

        return converted_batch

    def peek(self):
        """Grab the next batch from the dataloader
        without removing it from the queue"""
        return self.convert_batch(self._peek_next_batch())

    def convert_batch(self, batch, _to_dlpack_fn=None, _from_dlpack_fn=None):
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
        column_type = TorchColumn
        inputs, targets = batch

        tf_inputs = {}
        if inputs is not None:
            inputs_table = TensorTable(inputs)

            for col_name, col in inputs_table.items():
                tf_inputs[col_name] = convert_col(col, column_type, _to_dlpack_fn, _from_dlpack_fn)

        tf_target = None
        if targets is not None:
            if isinstance(targets, dict):
                targets_table = TensorTable(targets)

                tf_targets = {}
                for col_name, col in targets_table.items():
                    tf_targets[col_name] = convert_col(
                        col, column_type, _to_dlpack_fn, _from_dlpack_fn
                    )
                    tf_target = TensorTable(tf_targets).to_dict()
            else:
                targets_col = TensorColumn(targets)
                tf_target = convert_col(targets_col, column_type).values

        return (TensorTable(tf_inputs).to_dict(), tf_target)

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self
