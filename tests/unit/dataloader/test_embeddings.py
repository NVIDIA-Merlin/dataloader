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
import glob

import numpy as np
import pandas as pd
import pytest

from merlin.core.compat import cupy
from merlin.core.dispatch import HAS_GPU
from merlin.dataloader.loader_base import LoaderBase as Loader  # noqa
from merlin.dataloader.ops.embeddings import EmbeddingOperator
from merlin.dataloader.ops.padding import Padding
from merlin.io import Dataset
from merlin.schema import Tags
from merlin.schema.tags import TagSet
from merlin.table import TensorColumn, TensorTable


@pytest.mark.parametrize("unknown_value", [0, 1, np.random.uniform(size=10)])
def test_embedding_lookup_with_unknown_value(unknown_value):
    ids = np.array(["a", "b", "c"])
    embeddings = np.random.rand(3, 10)
    df = pd.DataFrame(
        {
            "id": ["a", "unknown"],
            "feature": [1, 2],
        }
    )

    dataset = Dataset(df, cpu=True)

    data_loader = Loader(
        dataset,
        batch_size=3,
        transforms=[
            EmbeddingOperator(
                embeddings,
                lookup_key="id",
                embedding_name="id_embedding",
                id_lookup_table=ids,
                unknown_value=unknown_value,
            ),
        ],
        shuffle=False,
    )
    x, y = data_loader.peek()

    assert x["id"].values.shape == (2,)
    embedding_values = x["id_embedding"].values
    if cupy and isinstance(embedding_values, cupy.ndarray):
        embedding_values = embedding_values.get()
    assert embedding_values.shape == (2, 10)
    np.testing.assert_equal(embedding_values[0], embeddings[0])
    np.testing.assert_equal(embedding_values[1], unknown_value)
    assert data_loader.output_schema.column_names == ["id", "feature", "id_embedding"]


def test_embedding_with_target():
    id_embeddings = np.random.rand(1000, 10)
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "target": [0, 1, 1, 0, 1, 0],
        }
    )

    dataset = Dataset(df)
    dataset.schema["target"] = dataset.schema["target"].with_tags(Tags.TARGET)

    data_loader = Loader(
        dataset,
        batch_size=2,
        transforms=[
            EmbeddingOperator(
                id_embeddings,
                lookup_key="id",
                embedding_name="id_embedding",
            ),
        ],
        shuffle=False,
    )
    x, y = data_loader.peek()
    assert x["id"].values.shape == (2,)
    assert x["id_embedding"].values.shape == (2, 10)
    assert data_loader.output_schema.column_names == ["id", "id_embedding"]
    assert y.tolist() == [0, 1]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_np_mmap_dl_no_lookup(tmpdir, embedding_ids, np_embeddings_from_pq, cpu):
    batch_size = 10000
    embeddings_file, _ = np_embeddings_from_pq
    cat_names = ["id"]
    embeddings = np.load(embeddings_file)
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.ITEM, Tags.ID])
    dataset.schema = schema
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(embeddings_file, mmap=True)],
        shuffle=False,
        device=cpu,
    )

    assert data_loader.input_schema.column_names == ["id"]
    assert data_loader.output_schema.column_names == ["id", "embeddings"]

    embeddings_dim = 1024
    embedding_schema = data_loader.output_schema["embeddings"]
    embeddings_value_count = embedding_schema.value_count
    assert embeddings_value_count.min == embeddings_value_count.max == embeddings_dim
    assert embedding_schema.tags == TagSet([Tags.EMBEDDING, Tags.ITEM])

    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + int(batch[0]["id"].shape[0])
        embeddings_vals = batch[0]["embeddings"].cpu().values
        assert (embeddings_vals == embeddings[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_np_mmap_dl_with_lookup(tmpdir, rev_embedding_ids, np_embeddings_from_pq, cpu):
    batch_size = 10000
    embeddings_file, id_lookup_file = np_embeddings_from_pq
    cat_names = ["id"]
    embedding_ids = rev_embedding_ids
    embeddings = np.load(embeddings_file)
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema

    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(embeddings_file, id_lookup_table=id_lookup_file, mmap=True)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + int(batch[0]["id"].shape[0])
        embeddings_vals = batch[0]["embeddings"].cpu().values
        assert (embeddings_vals == embeddings[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_np_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(embeddings_np)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + int(batch[0]["id"].shape[0])
        embeddings_vals = batch[0]["embeddings"].cpu().values
        assert (embeddings_vals == embeddings_np[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_np_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids = rev_embedding_ids
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(embeddings_np, id_lookup_table=embedding_ids.to_numpy())],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + int(batch[0]["id"].shape[0])
        embeddings_vals = batch[0]["embeddings"].cpu().values
        assert (embeddings_vals == embeddings_np[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_np_dl_with_lookup_ragged(
    tmpdir, rev_embedding_ids, embeddings_from_dataframe, cpu
):
    cat_names = ["id"]
    batch_size = 5
    pq_path = tmpdir / "id.parquet"
    embedding_ids = rev_embedding_ids["id"][:100].to_numpy()
    offsets = np.array([0, 10, 15, 20, 30, 40, 45, 55, 65, 75, 80, 90, 100])
    tensor_df = TensorTable({"id": TensorColumn(embedding_ids, offsets=offsets)}).to_df()
    tensor_df.to_parquet(pq_path)
    dataset = Dataset(str(pq_path), cpu=bool(cpu))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:100, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(embeddings_np, id_lookup_table=embedding_ids)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    old_end = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = old_end
        end = start + int(batch[0]["id"].cpu().values.shape[0])
        old_end = end
        id_offsets = batch[0]["id"].cpu().offsets
        embeddings_vals = batch[0]["embeddings"].cpu().values
        embeddings_offs = batch[0]["embeddings"].cpu().offsets
        assert (embeddings_vals == embeddings_np[start:end]).all()
        assert (embeddings_offs == id_offsets).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == offsets.shape[0] - 1


def test_embedding_with_padding():
    max_length = 10
    batch_size = 3
    id_embeddings = np.random.rand(1000, 16)
    df = pd.DataFrame(
        {
            "id": [[0, 1, 2], [3, 4], [5, 6, 7, 8]],
        }
    )

    dataset = Dataset(df)
    transform = (
        ["id"]
        >> Padding(padding_size=max_length, padding_value=0)
        >> EmbeddingOperator(id_embeddings, lookup_key="id", embedding_name="id_embedding")
    )
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=transform,
        shuffle=False,
    )
    x, _ = data_loader.peek()
    assert x["id"].values.shape == (batch_size, max_length)
    assert x["id_embedding"].values.shape == (batch_size, max_length, 16)
    assert data_loader.output_schema.column_names == ["id", "id_embedding"]
