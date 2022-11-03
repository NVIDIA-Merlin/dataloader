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
import rmm

from merlin.io import Dataset
from merlin.loader.torch import Loader
from merlin.schema import Tags

torch = pytest.importorskip("torch")

from merlin.loader.ops.embeddings import (  # noqa
    Numpy_Mmap_TorchEmbedding,
    Numpy_TorchEmbeddingOperator,
    TorchEmbeddingOperator,
)


def test_embedding_torch_np_mmap_dl_with_lookup(tmpdir, rev_embedding_ids, np_embeddings_from_pq):
    batch_size = 10000
    embeddings_file, lookup_file = np_embeddings_from_pq
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    embeddings = np.load(embeddings_file)
    pq_path = tmpdir / "id.parquet"
    rev_embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[Numpy_Mmap_TorchEmbedding(embeddings_file, ids_lookup_npz=lookup_file)],
        shuffle=False,
        # device="cpu",
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == rev_embedding_ids.shape[0]


def test_embedding_torch_np_mmap_dl_no_lookup(tmpdir, embedding_ids, np_embeddings_from_pq):
    batch_size = 10000
    embeddings_file, lookup_file = np_embeddings_from_pq
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    embeddings = np.load(embeddings_file)
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema

    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[Numpy_Mmap_TorchEmbedding(embeddings_file)],
        shuffle=False,
        # device="cpu",
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


def test_embedding_torch_np_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    batch_size = 10000
    embedding_ids = rev_embedding_ids
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_df = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[
            Numpy_TorchEmbeddingOperator(embeddings_df, id_lookup_table=embedding_ids.to_numpy())
        ],
        shuffle=False,
        # device="cpu",
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings_df[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


def test_embedding_torch_np_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    embeddings_df = embeddings_ds.to_ddf().compute().to_numpy()[:, 1:]
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[Numpy_TorchEmbeddingOperator(embeddings_df)],
        shuffle=False,
        # device="cpu",
    )
    full_len = 0
    for idx, batch in enumerate(data_loader):
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        start = idx * batch_size
        end = start + batch[0]["id"].shape[0]
        assert (batch[0]["embeddings"].cpu().numpy() == embeddings_df[start:end]).all()
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]


def test_embedding_torch_dl_with_lookup(tmpdir, rev_embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids = rev_embedding_ids
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy().astype("float32")
    torch_tensor = torch.from_numpy(np_tensor)
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TorchEmbeddingOperator(torch_tensor, id_lookup_table=embedding_ids.numpy())],
        shuffle=False,
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


def test_embedding_torch_dl_no_lookup(tmpdir, embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]
    batch_size = 10000
    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(10)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    paths = sorted(glob.glob(f"{embeddings_from_dataframe}/*"))
    embeddings_ds = Dataset(paths)
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy().astype("float32")
    torch_tensor = torch.from_numpy(np_tensor)
    data_loader = Loader(
        dataset,
        batch_size=batch_size,
        transforms=[TorchEmbeddingOperator(torch_tensor)],
        shuffle=False,
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
