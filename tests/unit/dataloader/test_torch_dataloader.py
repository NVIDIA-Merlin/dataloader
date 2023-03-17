#
# Copyright (c) 2021, NVIDIA CORPORATION.
#

# Licensed under the Aeache License, Version 2.0 (the "License");
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
import os
import time

import numpy as np
import pandas as pd
import pytest
from conftest import assert_eq

from merlin.core import dispatch
from merlin.core.compat import HAS_GPU
from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.schema import Tags

pytestmark = pytest.mark.torch

# If pytorch isn't installed skip these tests. Note that the
# torch_dataloader import needs to happen after this line
torch = pytest.importorskip("torch")
import merlin.dataloader.torch as torch_dataloader  # noqa isort:skip


@pytest.mark.parametrize("shape", [(), (1,), (2,), (3, 4)])
@pytest.mark.parametrize("num_cols", [1, 2])
def test_fixed_column(shape, num_cols):
    num_rows = 4
    batch_size = 3
    df = make_df(
        {
            f"col{i}": dispatch.random_uniform(size=(num_rows, *shape)).tolist()
            for i in range(num_cols)
        }
    )

    dataset = Dataset(df)
    for col in dataset.schema:
        dataset.schema[col.name] = dataset.schema[col.name].with_shape((None, *shape))

    loader = torch_dataloader.Loader(dataset, batch_size=batch_size)
    batches = list(loader)

    for tensor in batches[0][0].values():
        assert tensor.shape == (batch_size, *shape)

    for tensor in batches[1][0].values():
        assert tensor.shape == (1, *shape)


def test_shuffling():
    num_rows = 10000
    batch_size = 10000

    df = pd.DataFrame({"a": np.asarray(range(num_rows)), "b": np.asarray([0] * num_rows)})

    ds = Dataset(df)
    ds.schema["b"] = ds.schema["b"].with_tags([Tags.TARGET])

    train_dataset = torch_dataloader.Loader(ds, batch_size=batch_size, shuffle=True)

    batch = next(train_dataset)

    first_batch = batch[0]["a"].cpu()
    in_order = torch.arange(0, batch_size)

    assert (first_batch != in_order).any()
    assert (torch.sort(first_batch.squeeze()).values == in_order).all()


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_torch_drp_reset(tmpdir, batch_size, drop_last, num_rows):
    df = make_df(
        {
            "cat1": [1] * num_rows,
            "cat2": [2] * num_rows,
            "cat3": [3] * num_rows,
            "label": [0] * num_rows,
            "cont3": [3.0] * num_rows,
            "cont2": [2.0] * num_rows,
            "cont1": [1.0] * num_rows,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)

    ds = Dataset([path], cpu=True)
    ds.schema["label"] = ds.schema["label"].with_tags(Tags.TARGET)

    dataloader = torch_dataloader.Loader(
        ds,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    all_len = len(dataloader) if drop_last else len(dataloader) - 1
    all_rows = 0
    df_cols = df.columns.to_list()
    for idx, chunk in enumerate(dataloader):
        all_rows += len(chunk[0]["cat1"])
        if idx < all_len:
            for col in df_cols:
                if col in chunk[0].keys():
                    # We test if the values did NOT change in the dataloader
                    # Each column has only one unique value
                    # We test that each value in chunk (output of dataloader)
                    # is equal to every value in dataframe
                    if dispatch.HAS_GPU:
                        assert (
                            np.expand_dims(chunk[0][col].cpu().numpy(), 1) == df[col].values_host
                        ).all()
                    else:
                        assert (
                            np.expand_dims(chunk[0][col].cpu().numpy(), 1) == df[col].values
                        ).all()

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


@pytest.mark.parametrize("batch", [0, 100, 1000])
def test_gpu_file_iterator_ds(df, dataset, batch):
    df_itr = make_df({})
    for data_gd in dataset.to_iter(columns=list(dataset.schema.column_names)):
        df_itr = dispatch.concat([df_itr, data_gd], axis=0) if df_itr is not None else data_gd

    assert_eq(df_itr.reset_index(drop=True), df.reset_index(drop=True))


json_sample = {
    "conts": {
        "cont_1": {"dtype": np.float, "min_val": 0, "max_val": 1},
        "cont_2": {"dtype": np.float, "min_val": 0, "max_val": 1},
        "cont_3": {"dtype": np.float, "min_val": 0, "max_val": 1},
    },
    "cats": {
        "cat_1": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
            "multi_min": 2,
            "multi_max": 5,
            "multi_avg": 3,
        },
        "cat_5": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
            "multi_min": 2,
            "multi_max": 5,
            "multi_avg": 3,
        },
        "cat_2": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
        },
        "cat_3": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
        },
        "cat_4": {
            "dtype": None,
            "cardinality": 50,
            "min_entry_size": 1,
            "max_entry_size": 5,
        },
    },
    "labels": {"lab_1": {"dtype": None, "cardinality": 2}},
}


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.06])
@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("cpu", [False, True] if HAS_GPU else [True])
def test_dataloader_break(dataset, batch_size, part_mem_fraction, cpu):
    dataloader = torch_dataloader.Loader(
        dataset,
        batch_size=batch_size,
    )
    len_dl = len(dataloader) - 1

    first_chunk = 0
    idx = 0
    for idx, chunk in enumerate(dataloader):
        if idx == 0:
            first_chunk = len(chunk[0])
        if idx == 1:
            break
        del chunk

    dataloader.stop()

    assert idx < len_dl

    first_chunk_2 = 0
    for idx, chunk in enumerate(dataloader):
        if idx == 0:
            first_chunk_2 = len(chunk[0])
        del chunk
    assert idx == len_dl

    assert first_chunk == first_chunk_2


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.06])
@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("cpu", [False, True] if HAS_GPU else [True])
def test_dataloader(df, dataset, batch_size, part_mem_fraction, cpu):
    dataloader = torch_dataloader.Loader(
        dataset,
        batch_size=batch_size,
    )

    num_rows = df.shape[0]
    rows = 0

    # works with iterator alone, needs to test inside torch dataloader
    for idx, chunk in enumerate(dataloader):
        assert df["name-cat"].iloc[rows] == chunk[0]["name-cat"][0]
        rows += len(chunk[0]["x"])

    # accounts for incomplete batches at the end of chunks
    # that dont necesssarily have the full batch_size
    assert rows == num_rows

    def gen_col(batch):
        batch = batch[0]
        return (batch[0]), batch[1]

    t_dl = torch_dataloader.DLDataLoader(
        dataloader, collate_fn=gen_col, pin_memory=False, num_workers=0
    )
    rows = 0
    for idx, chunk in enumerate(t_dl):
        assert df["name-cat"].iloc[rows] == chunk[0]["name-cat"][0]
        rows += len(chunk[0]["x"])


@pytest.mark.parametrize("part_mem_fraction", [0.001, 0.1])
def test_kill_dl(dataset, part_mem_fraction):
    dataloader = torch_dataloader.Loader(dataset)

    results = {}

    for batch_size in [2**i for i in range(9, 25, 1)]:
        print("Checking batch size: ", batch_size)
        num_iter = max(10 * 1000 * 1000 // batch_size, 100)  # load 10e7 samples

        dataloader.batch_size = batch_size
        start = time.time()
        i = 0
        for i, data in enumerate(dataloader):
            if i >= num_iter:
                break
            del data

        stop = time.time()

        throughput = i * batch_size / (stop - start)
        results[batch_size] = throughput
        print(
            "batch size: ",
            batch_size,
            ", throughput: ",
            throughput,
            "items",
            i * batch_size,
            "time",
            stop - start,
        )


def test_mh_support(multihot_dataset):
    idx = 0
    for batch in torch_dataloader.Loader(multihot_dataset):
        idx = idx + 1
        cats_conts, labels = batch
        assert "Reviewers__values" in cats_conts
        assert "Reviewers__offsets" in cats_conts
        assert "Authors__values" in cats_conts
        assert "Authors__offsets" in cats_conts
    assert idx > 0


@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("cpu", [False, True] if HAS_GPU else [True])
def test_dataloader_schema(df, dataset, batch_size, cpu):
    with torch_dataloader.Loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    ) as data_loader:
        X, y = data_loader.peek()
    columns = set(dataset.schema.column_names) - {"label"}
    assert columns == set(X.keys())

    num_label_cols = y.shape[1] if len(y.shape) > 1 else 1
    assert num_label_cols == 1


def test_torch_map(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "sample_weight": [1.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)
    ds = Dataset(df)
    ds.schema["label"] = ds.schema["label"].with_tags(Tags.TARGET)

    def add_sample_weight(features, labels, sample_weight_col_name="sample_weight"):
        sample_weight = features.pop(sample_weight_col_name)

        return features, labels, sample_weight

    data_itr = torch_dataloader.Loader(
        ds,
        batch_size=10,
        shuffle=False,
    ).map(add_sample_weight)

    for X, y, sample_weight in data_itr:
        assert list(X["cat1"].cpu().numpy()) == [1] * 10
        assert list(X["cat2"].cpu().numpy()) == [2] * 10
        assert list(X["cat3"].cpu().numpy()) == [3] * 10
        assert list(X["cont1"].cpu().numpy()) == [1.0] * 10
        assert list(X["cont2"].cpu().numpy()) == [2.0] * 10

        assert list(sample_weight.cpu().numpy()) == [1.0] * 10


def test_1d_tensors():
    # create small dataset, add values to sparse_list
    df = make_df(
        {
            "cat1": [1] * 4,
            "cat2": [2] * 4,
            "cat3": [3] * 4,
            "cont2": [2.0] * 4,
            "spar1": [[1, 2, 3, 4], [4, 2, 4, 4], [1, 3, 4, 3], [1, 1, 3, 3]],
            "spar2": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14], [15, 16]],
        }
    )
    batch_size = 2

    ds = Dataset(df)
    dataloader = torch_dataloader.Loader(ds, batch_size=batch_size)
    for batch in dataloader:
        feats, labs = batch
        for col in feats.keys():
            feature_tensor = feats[col]
            assert len(list(feature_tensor.shape)) == 1


def test_offsets():
    # create small dataset, add values to sparse_list
    df = make_df(
        {
            "cat1": [1] * 4,
            "cat2": [1] * 4,
            "cont1": [2.0] * 4,
            "spar1": [[1, 2, 3, 4], [4, 2, 4, 4], [1, 3, 4, 3], [1, 1, 3, 3]],
            "spar2": [[1, 2, 3, 4, 5], [], [11, 0, -1, 14], [15, 16]],
        }
    )
    results = [
        {
            "spar1__values": [1, 2, 3, 4, 4, 2, 4, 4],
            "spar1__offsets": [0, 4, 8],
            "spar2__values": [1, 2, 3, 4, 5],
            "spar2__offsets": [0, 5, 5],
        },
        {
            "spar1__values": [1, 3, 4, 3, 1, 1, 3, 3],
            "spar1__offsets": [0, 4, 8],
            "spar2__values": [11, 0, -1, 14, 15, 16],
            "spar2__offsets": [0, 4, 6],
        },
    ]
    batch_size = 2
    ds = Dataset(df)
    dataloader = torch_dataloader.Loader(ds, batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(dataloader):
        feats, labs = batch
        for col in feats.keys():
            if col in results[i]:
                feature_tensor = list(feats[col].cpu().numpy())
                feature_result = results[i][col]
                assert feature_tensor == feature_result
