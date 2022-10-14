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

import pytest
import rmm

from merlin.io import Dataset
from merlin.loader.tensorflow import Loader
from merlin.schema import Tags

tf = pytest.importorskip("tensorflow")

from merlin.loader.ops.embeddings import (  # noqa
    Numpy_Mmap_TFEmbedding,
    Numpy_TFEmbeddingOperator,
    TFEmbeddingOperator,
)


def test_embedding_tf_np_mmap_dl(tmpdir, embedding_ids, np_embeddings_from_pq):
    embeddings_file, _ = np_embeddings_from_pq
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]

    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
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
        batch_size=100000,
        transforms=[Numpy_Mmap_TFEmbedding("/raid/data/embeds/embeddings.npy")],
        shuffle=False,
    )
    full_len = 0
    for batch in data_loader:
        assert "embeddings" in batch[0]
        assert batch[0]["embeddings"][0].shape[1] == 1024
        assert "id" in batch[0]
        full_len += batch[0]["id"].shape[0]
    assert full_len == embedding_ids.shape[0]


def test_embedding_tf_np_dl(tmpdir, embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]

    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
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
    data_loader = Loader(
        dataset,
        batch_size=100000,
        transforms=[Numpy_TFEmbeddingOperator(embeddings_ds.to_ddf().compute().to_numpy())],
        shuffle=False,
    )
    full_len = 0
    for batch in data_loader:
        assert "embeddings" in batch[0]
        assert batch[0]["embeddings"][0].shape[0] == 1024
        assert "id" in batch[0]
        full_len += batch[0]["id"].shape[0]
    assert full_len == embedding_ids.shape[0]


def test_embedding_tf_dl(tmpdir, embedding_ids, embeddings_from_dataframe):
    rmm.reinitialize(managed_memory=True)
    cat_names = ["id"]

    pq_path = tmpdir / "id.parquet"
    embedding_ids.to_parquet(pq_path, row_group_size_bytes=134217728)
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
    np_tensor = embeddings_ds.to_ddf().compute().to_numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor[:, 1:])
    data_loader = Loader(
        dataset,
        batch_size=100000,
        transforms=[TFEmbeddingOperator(tf_tensor)],
        shuffle=False,
    )
    full_len = 0
    for batch in data_loader:
        assert "embeddings" in batch[0]
        assert batch[0]["embeddings"][0].shape[1] == 1024
        assert "id" in batch[0]
        full_len += batch[0]["id"].shape[0]
    assert full_len == embedding_ids.shape[0]
