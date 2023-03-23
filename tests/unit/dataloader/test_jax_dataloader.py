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

import numpy as np
import pytest

from merlin.core.compat import HAS_GPU
from merlin.core.dispatch import make_df, random_uniform
from merlin.io import Dataset
from merlin.schema import Tags

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jax_dataloader = pytest.importorskip("merlin.dataloader.jax")


@pytest.mark.parametrize("element_shape", [(), (1,), (2,), (3, 4)])
@pytest.mark.parametrize("num_cols", [1, 2])
def test_fixed_column(element_shape, num_cols):
    num_rows = 4
    batch_size = 3
    df = make_df(
        {
            f"col{i}": random_uniform(size=(num_rows, *element_shape)).tolist()
            for i in range(num_cols)
        }
    )

    dataset = Dataset(df)
    for col in dataset.schema:
        dataset.schema[col.name] = dataset.schema[col.name].with_shape((None, *element_shape))

    loader = jax_dataloader.Loader(dataset, batch_size=batch_size)
    batches = list(loader)

    for tensor in batches[0][0].values():
        assert tensor.shape == (batch_size, *element_shape)

    for tensor in batches[1][0].values():
        assert tensor.shape == (1, *element_shape)


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("cpu", [True, False] if HAS_GPU else [True])
def test_dataloader_schema(tmpdir, dataset, cpu, num_rows):
    df = make_df(
        {
            "cat1": np.ones(num_rows, dtype=np.int32),
            "cat2": 2 * np.ones(num_rows, dtype=np.int32),
            "cat3": 3 * np.ones(num_rows, dtype=np.int32),
            "label": np.zeros(num_rows, dtype=np.int32),
            "cont1": np.ones(num_rows, dtype=np.float32),
            "cont2": 2 * np.ones(num_rows, dtype=np.float32),
        }
    )
    dataset = Dataset(df, use_cpu=cpu)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET)

    with jax_dataloader.Loader(
        dataset,
        batch_size=num_rows,
        shuffle=False,
    ) as data_loader:
        inputs, target = data_loader.peek()
    columns = set(dataset.schema.column_names) - {"label"}
    assert set(inputs) == columns
