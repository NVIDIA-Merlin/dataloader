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
# import glob
import numpy as np
import pytest
import rmm

from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.loader.tensorflow import Loader
from merlin.schema import Tags

pytest.importorskip("tensorflow")


@pytest.mark.parametrize("engine", ["parquet"])
def test_embedding_workflow_dl(datasets, engine, tmpdir):
    # df_lib = "pd"
    num_ids = 2
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=1024 * 1024 * 1024,
        devices=[0, 1],
        managed_memory=True,
    )
    # rmm.reinitialize(managed_memory=True)
    # cont_names = ["x", "y"]
    cat_names = ["id"]

    df = make_df({"id": np.arange(num_ids * 10e5).astype("int32")})
    pq_path = tmpdir / "id.parquet"
    # os.makedirs(str(pq_path), exist_ok=True)
    df.to_parquet(str(pq_path), row_group_size_bytes=134217728)
    dataset = Dataset(str(pq_path))
    dataset = dataset.repartition(100000)
    schema = dataset.schema
    for col_name in cat_names:
        schema[col_name] = schema[col_name].with_tags(Tags.CATEGORICAL)
    dataset.schema = schema
    # df = concat([df,df,df,df])
    # for idx, x in enumerate(range(0,len(df), 100000000)):
    # df.iloc[x:x + 100000000].to_parquet(f"/raid/data/embeds/ids/data_{idx}.parquet")

    # df_ext = make_df(df["id"].unique().copy())

    # for idx, splt in enumerate(np.array_split(df.to_numpy(), num_ids)):
    #     indexes = make_series(list(range(1024, 1024 * (splt.shape[0] + 2), 1024))).astype("int32")
    #     embeddings_col = build_cudf_list_column(
    #               np.random.rand(1024 * (indexes.shape[0] + 1)), indexes
    #           )
    #     embeddings_series = make_series(embeddings_col)
    #     df = make_df({"id": np.squeeze(splt).astype('int32'), "embeddings": embeddings_series})

    # print(f"writing file {idx}")
    # df.to_parquet(f"/raid/data/embeds/{idx}.parquet")

    # breakpoint()
    # paths = sorted(glob.glob("/raid/data/embeds/*"))[:num_ids]
    # ds = Dataset(paths[:10], device="cpu")
    # df_ext = ds.to_ddf().compute()
    # if df_lib == "pd":
    #     df_list = [pd.read_parquet(path) for path in paths]
    #     breakpoint()
    #     df_ext = pd.concat(df_list)
    # else:
    #     df_list = [cudf.read_parquet(path) for path in paths]
    #     df_ext = cudf.concat(df_list, copy=False)

    # df_ext["embeddings"] = embeddings_series

    # wkflow = nvt.Workflow(joined)
    # target_dataset = wkflow.fit_transform(dataset)

    data_loader = Loader(
        dataset,
        batch_size=10000,
        transform=[],
        shuffle=False,
    )

    full_len = 0
    for batch in data_loader:
        # breakpoint()
        assert "embeddings" in batch[0]
        assert "id" in batch[0]
        embeddings_index_size = batch[0]["embeddings"][1].shape[0]
        assert embeddings_index_size == batch[0]["id"].shape[0]
        full_len += embeddings_index_size
    assert full_len == df.shape[0]
