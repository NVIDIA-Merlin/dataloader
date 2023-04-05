#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import pytest

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.dataloader import ArrayLoader as Loader
from merlin.dataloader.ops.padding import Padding
from merlin.io import Dataset


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_padding(cpu):
    padding_size = 5
    padding_value = 0
    batch_size = 3
    df = make_df({"a": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]})
    dataset = Dataset(df, cpu=bool(cpu))
    dl_graph = ["a"] >> Padding(padding_size, padding_value)
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=dl_graph,
        shuffle=False,
        device=cpu,
    )
    col_schema = data_loader._output_schema["a"]
    assert col_schema.shape.as_tuple[-1] == padding_size
    assert not col_schema.shape.is_fixed  # because we don't know the size of the final batch
    assert not col_schema.is_ragged
    for batch in data_loader:
        column = batch[0]["a"]
        assert column.values.shape[-1] == padding_size
        assert column.offsets is None
