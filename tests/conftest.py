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
import gc
import glob
import random
from unittest.mock import patch

import dask
import pandas as pd
from npy_append_array import NpyAppendArray

from merlin.core.compat import cudf
from merlin.core.compat import numpy as np
from merlin.dataloader.loader_base import LoaderBase

if cudf:
    try:
        import cudf.testing._utils

        assert_eq = cudf.testing._utils.assert_eq
    except ImportError:
        import cudf.tests.utils

        assert_eq = cudf.tests.utils.assert_eq
else:

    def assert_eq(a, b, *args, **kwargs):
        if isinstance(a, pd.DataFrame):
            return pd.testing.assert_frame_equal(a, b, *args, **kwargs)
        elif isinstance(a, pd.Series):
            return pd.testing.assert_series_equal(a, b, *args, **kwargs)
        else:
            return np.testing.assert_allclose(a, b)


import pytest

from merlin.core.dispatch import concat_columns, get_lib, make_df
from merlin.io import Dataset
from merlin.schema import Tags


@pytest.fixture(scope="session")
def df():
    df = (
        dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-01-04",
            freq="60s",
            dtypes={
                "name-cat": str,
                "name-string": str,
                "id": int,
                "label": int,
                "x": float,
                "y": float,
                "z": float,
            },
        )
        .reset_index()
        .compute()
    )

    # convert to a categorical
    df["name-string"] = df["name-string"].astype("category").cat.codes.astype("int32")
    df["name-cat"] = df["name-cat"].astype("category").cat.codes.astype("int32")

    # Add two random null values to each column
    imax = len(df) - 1
    for col in df.columns:
        if col in ["name-cat", "label", "id"]:
            break
        for _ in range(2):
            rand_idx = random.randint(1, imax - 1)
            if rand_idx == df[col].shape[0] // 2:
                # dont want null in median
                rand_idx += 1
            df[col].iloc[rand_idx] = None

    df = df.drop("timestamp", axis=1)
    return df


@pytest.fixture(scope="session")
def parquet_path(df, tmpdir_factory):
    # we're writing out to parquet independently of the dataset here
    # since the serialization is relatively expensive, and we can do at the 'session'
    # scope - but the dataset fixture needs to be at the 'function' scope so that
    # we can set options like 'part_mem_fraction' and 'cpu'
    datadir = tmpdir_factory.mktemp("test_dataset")
    half = int(len(df) // 2)

    # Write Parquet Dataset
    df.iloc[:half].to_parquet(str(datadir / "dataset-0.parquet"), chunk_size=1000)
    df.iloc[half:].to_parquet(str(datadir / "dataset-1.parquet"), chunk_size=1000)
    return datadir


@pytest.fixture(scope="function")
def dataset(request, tmpdir_factory, parquet_path):
    try:
        gpu_memory_frac = request.getfixturevalue("gpu_memory_frac")
    except Exception:  # pylint: disable=broad-except
        gpu_memory_frac = 0.01

    try:
        cpu = request.getfixturevalue("cpu")
    except Exception:  # pylint: disable=broad-except
        cpu = None

    dataset = Dataset(parquet_path, engine="parquet", part_mem_fraction=gpu_memory_frac, cpu=cpu)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET)
    return dataset


# Allow to pass devices as parameters
def pytest_addoption(parser):
    parser.addoption("--devices", action="store", default="0", help="0,1,..,n-1")
    parser.addoption("--report", action="store", default="1", help="0 | 1")


@pytest.fixture
def devices(request):
    return request.config.getoption("--devices")


@pytest.fixture
def report(request):
    return request.config.getoption("--report")


@pytest.fixture
def multihot_data():
    return {
        "Authors": [[0], [0, 5], [1, 2], [2]],
        "Reviewers": [[0], [0, 5], [1, 2], [2]],
        "Engaging User": [1, 1, 0, 4],
        "Embedding": [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.8, 0.4, 0.2],
        ],
        "Post": [1, 2, 3, 4],
    }


@pytest.fixture
def multihot_dataset(multihot_data):
    ds = Dataset(make_df(multihot_data))
    ds.schema["Post"] = ds.schema["Post"].with_tags(Tags.TARGET)
    return ds


@pytest.fixture(scope="session")
def num_embedding_ids():
    return 1


@pytest.fixture(scope="session")
def embeddings_part_size():
    return 1e5


@pytest.fixture(scope="session")
def embedding_ids(num_embedding_ids, embeddings_part_size):
    df = make_df({"id": np.arange(num_embedding_ids * embeddings_part_size).astype("int32")})
    return df


@pytest.fixture(scope="session")
def rev_embedding_ids(embedding_ids, tmpdir_factory):
    df_rev = embedding_ids["id"][::-1]
    df_rev.reset_index(inplace=True, drop=True)
    return make_df(df_rev)


@pytest.fixture(scope="session")
def embeddings_from_dataframe(embedding_ids, num_embedding_ids, tmpdir_factory):
    embed_dir = tmpdir_factory.mktemp("embeds")
    for idx, splt in enumerate(np.array_split(embedding_ids.to_numpy(), num_embedding_ids)):
        vals = make_df(np.random.rand(splt.shape[0], 1024))
        ids = make_df({"id": np.squeeze(splt)})
        full = concat_columns([ids, vals])
        full.columns = [str(col) for col in full.columns]
        full.to_parquet(f"{embed_dir}/{idx}.parquet")
    return embed_dir


@pytest.fixture(scope="session")
def rev_embeddings_from_dataframe(rev_embedding_ids, num_embedding_ids, tmpdir_factory):
    embed_dir = tmpdir_factory.mktemp("rev_embeds")
    for idx, splt in enumerate(np.array_split(rev_embedding_ids.to_numpy(), num_embedding_ids)):
        vals = make_df(np.random.rand(splt.shape[0], 1024))
        ids = make_df({"id": np.squeeze(splt)})
        full = concat_columns([ids, vals])
        full.columns = [str(col) for col in full.columns]
        full.to_parquet(f"{embed_dir}/{idx}.parquet")
    return embed_dir


def build_embeddings_from_pq(
    df_paths, embedding_filename="embeddings.npy", lookup_filename="lookup_ids"
):
    df_lib = get_lib()
    with NpyAppendArray(embedding_filename) as nf:
        with NpyAppendArray(lookup_filename) as lf:
            for path in df_paths:
                rows = df_lib.read_parquet(path)
                numpy_rows = rows.to_numpy()
                indices = np.ascontiguousarray(numpy_rows[:, 0])
                vectors = np.ascontiguousarray(numpy_rows[:, 1:])
                lf.append(indices)
                nf.append(vectors)
                del rows
                del numpy_rows
                del indices
                del vectors
                gc.collect()
    return embedding_filename, lookup_filename


@pytest.fixture(scope="session")
def np_embeddings_from_pq(rev_embeddings_from_dataframe, tmpdir_factory):
    paths = sorted(glob.glob(f"{rev_embeddings_from_dataframe}/*"))
    embed_dir = tmpdir_factory.mktemp("np_embeds")
    embeddings_file = f"{embed_dir}/embeddings.npy"
    lookup_ids_file = f"{embed_dir}/ids_lookup.npy"
    npy_filename, lookup_filename = build_embeddings_from_pq(
        paths, embeddings_file, lookup_ids_file
    )
    return npy_filename, lookup_filename


@pytest.fixture(scope="function", autouse=True)
def cleanup_dataloader():
    """After each test runs. Call .stop() on any dataloaders created during the test.
    The avoids issues with background threads hanging around and interfering with subsequent tests.
    This happens when a dataloader is partially consumed (not all batches are iterated through).
    """
    with patch.object(
        LoaderBase, "__iter__", side_effect=LoaderBase.__iter__, autospec=True
    ) as patched:
        yield
        for call in patched.call_args_list:
            call.args[0].stop()
