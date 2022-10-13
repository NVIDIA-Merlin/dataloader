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
import pytest
import rmm

from merlin.io import Dataset
from merlin.loader.torch import Loader
from merlin.schema import Tags

torch = pytest.importorskip("torch")

from merlin.loader.ops.embeddings import Numpy_Mmap_TorchEmbedding  # noqa


def test_embedding_torch_dl(tmpdir, embedding_ids, np_embeddings_from_pq):
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

    data_loader = Loader(
        dataset,
        batch_size=100000,
        transforms=[Numpy_Mmap_TorchEmbedding(embeddings_file)],
        shuffle=False,
        # device="cpu",
    )
    full_len = 0
    for batch in data_loader:
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        full_len += batch[0]["embeddings"].shape[0]
    assert full_len == embedding_ids.shape[0]
