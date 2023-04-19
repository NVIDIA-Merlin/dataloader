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

from merlin.core.dispatch import HAS_GPU
from merlin.dataloader.loader_base import LoaderBase as Loader  # noqa
from merlin.dataloader.ops.embeddings import (  # noqa
    EmbeddingOperator,
    MmapNumpyEmbedding,
    NumpyEmbeddingOperator,
)
from merlin.io import Dataset
from merlin.schema import Tags


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
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[MmapNumpyEmbedding(embeddings_file)],
        shuffle=False,
        device=cpu,
    )

    assert data_loader.input_schema.column_names == ["id"]
    assert data_loader.output_schema.column_names == ["id", "embeddings"]

    embeddings_dim = 1024
    embeddings_value_count = data_loader.output_schema["embeddings"].value_count
    assert embeddings_value_count.min == embeddings_value_count.max == embeddings_dim

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

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[MmapNumpyEmbedding(embeddings_file, ids_lookup_npz=id_lookup_file)],
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

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[NumpyEmbeddingOperator(embeddings_np)],
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

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[
            NumpyEmbeddingOperator(embeddings_np, id_lookup_table=embedding_ids.to_numpy())
        ],
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
def test_embedding_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe, cpu):
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

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(np_tensor)],
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
        assert (embeddings_vals == np_tensor[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe, cpu):
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

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags([Tags.CATEGORICAL, Tags.EMBEDDING])
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[EmbeddingOperator(np_tensor, id_lookup_table=embedding_ids.to_numpy())],
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
        assert (embeddings_vals == np_tensor[start:end]).all()
        full_len += int(batch[0]["embeddings"].shape[0])
    assert full_len == embedding_ids.shape[0]
