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
from functools import partial

from merlin.core.compat.torch import torch as th
from merlin.dataloader.loader_base import LoaderBase
from merlin.table import TensorColumn, TensorTable, TorchColumn
from merlin.table.conversions import _dispatch_dlpack_fns, convert_col


class Loader(LoaderBase, th.utils.data.IterableDataset):
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

        self.create_table = partial(TensorTable, _unsafe=True)
        self.create_column = partial(TensorColumn, _unsafe=True)
        column = self.create_column(self.array_lib().array([]))

        _to_dlpack_fn, _from_dlpack_fn = _dispatch_dlpack_fns(column, TorchColumn)
        self.convert_col = partial(
            convert_col, _to_dlpack_fn=_to_dlpack_fn, _from_dlpack_fn=_from_dlpack_fn, _unsafe=True
        )

    def __next__(self):
        """Get the next batch from the dataloader"""
        with self._get_device_ctx(self.device):
            converted_batch = self.convert_batch(super().__next__())
            for map_fn in self._map_fns:
                converted_batch = map_fn(*converted_batch)

        return converted_batch

    def peek(self):
        """Grab the next batch from the dataloader
        without removing it from the queue"""
        with self._get_device_ctx(self.device):
            converted_batch = self.convert_batch(self._peek_next_batch())
            for map_fn in self._map_fns:
                converted_batch = map_fn(*converted_batch)

        return converted_batch

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
        column_type = TorchColumn
        inputs, targets = batch
        torch_inputs = {}
        if inputs is not None:
            inputs_table = TensorTable(inputs, _unsafe=True)
            for col_name, col in inputs_table.items():
                torch_inputs[col_name] = self.convert_col(col, column_type)

        torch_targets = None
        if targets is not None:
            if isinstance(targets, dict):
                targets_table = TensorTable(targets, _unsafe=True)
                torch_targets = {}
                for col_name, col in targets_table.items():
                    torch_targets[col_name] = self.convert_col(col, column_type)
                torch_targets = TensorTable(torch_targets, _unsafe=True).to_dict()
            else:
                targets_col = TensorColumn(targets, _unsafe=True)
                torch_targets = self.convert_col(targets_col, column_type).values

        return (TensorTable(torch_inputs, _unsafe=True).to_dict(), torch_targets)

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self

    def _get_device_ctx(self, dev):
        if dev == "cpu" or not th:
            return contextlib.nullcontext()
        return th.cuda.device(f"cuda:{dev}")


class DLDataLoader(th.utils.data.DataLoader):
    """
    This class is an extension of the torch dataloader.
    It is required to support the FastAI framework.
    """

    @property
    def device(self):
        return th.device("cuda" if th.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset)
