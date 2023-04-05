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

from merlin.core.dispatch import HAS_GPU, make_df
from merlin.io import Dataset
from merlin.schema import Tags

arr_dataloader = pytest.importorskip("merlin.dataloader")


@pytest.mark.parametrize("num_rows", [1000, 10000])
@pytest.mark.parametrize("cpu", [True, False] if HAS_GPU else [True])
def test_dataloader(tmpdir, dataset, cpu, num_rows):
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
    dataset = Dataset(df, cpu=cpu)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET)

    data_loader = arr_dataloader.ArrayLoader(
        dataset, batch_size=int(num_rows / 10), shuffle=False, device="cpu" if cpu else "0"
    )
    inputs, target = data_loader.peek()
    columns = set(dataset.schema.column_names) - {"label"}
    assert set(inputs) == columns

    for idx, batch in enumerate(data_loader):
        for col in columns:
            start = idx * int(num_rows / 10)
            end = start + int(num_rows / 10)
            assert np.all(batch[0][col].ravel() == df[col].iloc[start:end])
