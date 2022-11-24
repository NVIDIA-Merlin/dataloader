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
import pytest

from merlin.core.dispatch import HAS_GPU
from merlin.io import Dataset
from merlin.schema import Tags

tf = pytest.importorskip("tensorflow")

from merlin.dataloader.ops.embeddings.tf_embedding_op import (  # noqa
    TF_MmapNumpyTorchEmbedding,
    TF_NumpyEmbeddingOperator,
    TFEmbeddingOperator,
)
from merlin.dataloader.tensorflow import Loader  # noqa


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_np_mmap_dl_no_lookup(tmpdir, embedding_ids, np_embeddings_from_pq, cpu):
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
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TF_MmapNumpyTorchEmbedding(embeddings_file)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_np_mmap_dl_with_lookup(tmpdir, rev_embedding_ids, np_embeddings_from_pq, cpu):
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
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TF_MmapNumpyTorchEmbedding(embeddings_file, ids_lookup_npz=id_lookup_file)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_np_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TF_NumpyEmbeddingOperator(embeddings_np)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings_np[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_np_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids = rev_embedding_ids
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_np = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[
            TF_NumpyEmbeddingOperator(embeddings_np, id_lookup_table=embedding_ids.to_numpy())
        ],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings_np[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    tf_tensor = tf.convert_to_tensor(np_tensor)
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TFEmbeddingOperator(tf_tensor)],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == np_tensor[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


@pytest.mark.parametrize("cpu", [None, "cpu"] if HAS_GPU else ["cpu"])
def test_embedding_tf_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe, cpu):
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids = rev_embedding_ids
    embedding_ids.to_parquet(pq_path)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    tf_tensor = tf.convert_to_tensor(np_tensor)
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TFEmbeddingOperator(tf_tensor, id_lookup_table=embedding_ids.to_numpy())],
        shuffle=False,
        device=cpu,
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == np_tensor[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]
