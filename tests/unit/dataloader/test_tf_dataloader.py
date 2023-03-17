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

import importlib.util
import os
import subprocess
import time
import timeit

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from merlin.core.compat import HAS_GPU, cupy
from merlin.core.dispatch import make_df, random_uniform
from merlin.io import Dataset
from merlin.schema import Tags

pytestmark = pytest.mark.tensorflow


tf = pytest.importorskip("tensorflow")
# If tensorflow isn't installed skip these tests. Note that the
# tf_dataloader import needs to happen after this line
tf_dataloader = pytest.importorskip("merlin.dataloader.tensorflow")


@pytest.mark.parametrize("shape", [(), (1,), (2,), (3, 4), (4, 5, 6)])
@pytest.mark.parametrize("num_cols", [1, 2])
def test_fixed_column(shape, num_cols):
    num_rows = 4
    batch_size = 3
    df = make_df(
        {f"col{i}": random_uniform(size=(num_rows, *shape)).tolist() for i in range(num_cols)}
    )

    dataset = Dataset(df)
    for col in dataset.schema:
        dataset.schema[col.name] = dataset.schema[col.name].with_shape((None, *shape))

    loader = tf_dataloader.Loader(dataset, batch_size=batch_size)
    batches = list(loader)

    for tensor in batches[0][0].values():
        assert tensor.shape == (batch_size, *shape)

    for tensor in batches[1][0].values():
        assert tensor.shape == (1, *shape)


def test_peek():
    df = make_df({"a": [1, 2, 3]})
    dataset = Dataset(df)
    with tf_dataloader.Loader(dataset, batch_size=1, shuffle=False) as loader:
        first_batch = loader.peek()
        all_batches = list(loader)
    test_case = tf.test.TestCase()
    test_case.assertAllEqual(first_batch, all_batches[0])
    assert len(all_batches) == 3


def test_set_input_schema():
    df = make_df({"a": [1, 2, 3], "b": [[4], [5, 6], [7]]})
    dataset = Dataset(df)
    loader = tf_dataloader.Loader(dataset, batch_size=1)
    loader.input_schema = dataset.schema.excluding_by_name(["b"])
    x, y = loader.peek()
    assert set(x.keys()) == {"a"}


def test_set_input_schema_after_start():
    df = make_df({"a": [1, 2, 3], "b": [4, 5, 6]})
    dataset = Dataset(df)
    loader = tf_dataloader.Loader(dataset, batch_size=1)
    with pytest.raises(RuntimeError) as exc_info:
        _ = next(loader)
        loader.input_schema = dataset.schema.excluding_by_name(["b"])
    assert "Setting the input_schema after the dataloader has started is not supported" in str(
        exc_info.value
    )


def test_simple_model():
    df = make_df({"a": [0.1, 0.2, 0.3], "label": [0, 1, 0]})
    dataset = Dataset(df)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET)

    loader = tf_dataloader.Loader(dataset, batch_size=1)

    inputs = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    outputs = tf.keras.layers.Dense(16, "relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="softmax")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])
    history_callback = model.fit(loader, epochs=2, shuffle=False)
    assert len(history_callback.history["loss"]) == 2
    assert all(loss > 0.0 for loss in history_callback.history["loss"])

    preds_model = model.predict({"a": tf.constant([0.1, 0.2, 0.3])})
    preds_loader = model.predict(loader)
    assert preds_model.shape == preds_loader.shape

    _ = model.evaluate(loader)


def test_with_device():
    dataset = Dataset(make_df({"a": [1]}))
    tf_dataloader.Loader(dataset, batch_size=1, device=1).peek()


def test_nested_list():
    num_rows = 100
    batch_size = 12

    df = pd.DataFrame(
        {
            # ToDo: We deprioritized nested columns - it requires multiple offsets
            # "data": [
            #   np.random.rand(np.random.randint(10) + 1, 3).tolist() for i in range(num_rows)
            # ],
            # keep a data field because we use index in the tests
            "data": [np.random.rand() for i in range(num_rows)],
            "data2": [np.random.rand(np.random.randint(10) + 1).tolist() for i in range(num_rows)],
            "label": [np.random.rand() for i in range(num_rows)],
        }
    )
    ds = Dataset(df)
    schema = ds.schema
    schema["label"] = schema["label"].with_tags([Tags.TARGET])
    ds.schema = schema
    loader = tf_dataloader.Loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(loader)
    # [[1,2,3],[3,1],[...],[]]
    #     @tf.function
    #     def _ragged_for_nested_data_col():
    #         nested_data_col = tf.RaggedTensor.from_row_lengths(
    #             batch[0]["data"][0][:, 0], tf.cast(batch[0]["data"][1][:, 0], tf.int32)
    #         ).to_tensor()
    #         return nested_data_col

    #     nested_data_col = _ragged_for_nested_data_col()
    #     true_data_col = tf.reshape(
    #         tf.ragged.constant(df.iloc[:batch_size, 0].tolist()).to_tensor(), [batch_size, -1]
    #     )

    # [1,2,3]
    @tf.function
    def _ragged_for_multihot_data_col():
        multihot_data2_col = tf.RaggedTensor.from_row_splits(
            batch[0]["data2__values"], tf.cast(batch[0]["data2__offsets"], tf.int32)
        ).to_tensor()
        return multihot_data2_col

    multihot_data2_col = _ragged_for_multihot_data_col()
    true_data2_col = tf.reshape(
        tf.ragged.constant(df.iloc[:batch_size, 1].tolist()).to_tensor(),
        [batch_size, -1],
    )
    #     assert nested_data_col.shape == true_data_col.shape
    #     assert np.allclose(nested_data_col.numpy(), true_data_col.numpy())
    assert multihot_data2_col.shape == true_data2_col.shape
    assert np.allclose(multihot_data2_col.numpy(), true_data2_col.numpy())


def test_shuffling():
    num_rows = 10000
    batch_size = 10000

    df = pd.DataFrame({"a": np.asarray(range(num_rows)), "b": np.asarray([0] * num_rows)})

    ds = Dataset(df)
    ds.schema["b"] = ds.schema["b"].with_tags([Tags.TARGET])

    train_dataset = tf_dataloader.Loader(ds, batch_size=batch_size, shuffle=True)

    batch = next(train_dataset)

    first_batch = tf.reshape(tf.cast(batch[0]["a"].cpu(), tf.int32), (batch_size,))
    in_order = tf.range(0, batch_size, dtype=tf.int32)

    assert (first_batch != in_order).numpy().any()
    assert (tf.sort(first_batch) == in_order).numpy().all()


@pytest.mark.parametrize("batch_size", [10, 9, 8])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("num_rows", [100])
def test_tf_drp_reset(tmpdir, batch_size, drop_last, num_rows):
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

    ds = Dataset(df)
    ds.schema["label"] = ds.schema["label"].with_tags(Tags.TARGET)

    data_itr = tf_dataloader.Loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
    )

    all_len = len(data_itr) if drop_last else len(data_itr) - 1
    all_rows = 0
    for idx, (X, y) in enumerate(data_itr):
        all_rows += len(X["cat1"])
        if idx < all_len:
            assert list(X["cat1"].numpy()) == [1] * batch_size
            assert list(X["cat2"].numpy()) == [2] * batch_size
            assert list(X["cat3"].numpy()) == [3] * batch_size
            assert list(X["cont1"].numpy()) == [1.0] * batch_size
            assert list(X["cont2"].numpy()) == [2.0] * batch_size
            assert list(X["cont3"].numpy()) == [3.0] * batch_size

    if drop_last and num_rows % batch_size > 0:
        assert num_rows > all_rows
    else:
        assert num_rows == all_rows


def test_tf_catname_ordering(tmpdir):
    df = make_df(
        {
            "cat1": [1] * 100,
            "cat2": [2] * 100,
            "cat3": [3] * 100,
            "label": [0] * 100,
            "cont3": [3.0] * 100,
            "cont2": [2.0] * 100,
            "cont1": [1.0] * 100,
        }
    )
    path = os.path.join(tmpdir, "dataset.parquet")
    df.to_parquet(path)

    ds = Dataset(df)
    ds.schema["label"] = ds.schema["label"].with_tags(Tags.TARGET)

    data_itr = tf_dataloader.Loader(
        ds,
        batch_size=10,
        shuffle=False,
    )

    for X, y in data_itr:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10
        assert list(X["cont3"].numpy()) == [3.0] * 10


def test_tf_map(tmpdir):
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
        sample_weight = tf.cast(features.pop(sample_weight_col_name) > 0, tf.float32)

        return features, labels, sample_weight

    data_itr = tf_dataloader.Loader(
        ds,
        batch_size=10,
        shuffle=False,
    ).map(add_sample_weight)

    for X, y, sample_weight in data_itr:
        assert list(X["cat1"].numpy()) == [1] * 10
        assert list(X["cat2"].numpy()) == [2] * 10
        assert list(X["cat3"].numpy()) == [3] * 10
        assert list(X["cont1"].numpy()) == [1.0] * 10
        assert list(X["cont2"].numpy()) == [2.0] * 10

        assert list(sample_weight.numpy()) == [1.0] * 10


# TODO: include use_columns option
# TODO: include parts_per_chunk test
@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.06])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("cpu", [False, True] if HAS_GPU else [True])
def test_tensorflow_dataloader(
    tmpdir,
    cpu,
    dataset,
    batch_size,
    gpu_memory_frac,
):
    dataloader = tf_dataloader.Loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    _ = tf.random.uniform((1,))

    rows = 0
    for idx in range(len(dataloader)):
        X, y = next(dataloader)

        # first elements to check epoch-to-epoch consistency
        if idx == 0:
            X0, y0 = X, y

        # check that we have at most batch_size elements
        num_samples = y.shape[0]
        if num_samples != batch_size:
            try:
                next(dataloader)
            except StopIteration:
                rows += num_samples
                continue
            else:
                raise ValueError(f"Batch size too small at idx {idx}")

        # check that all the features in X have the
        # appropriate length and that the set of
        # their names is exactly the set of names in
        # `columns`
        these_cols = set(dataset.schema.column_names)
        these_cols.remove("label")
        for column, x in X.items():
            try:
                these_cols.remove(column)
            except ValueError as e:
                raise AssertionError from e
            assert x.shape[0] == num_samples
        assert len(these_cols) == 0
        rows += num_samples

    assert (idx + 1) * batch_size >= rows
    row_count = 60 * 24 * 3
    assert rows == row_count
    # if num_samples is equal to batch size,
    # we didn't exhaust the iterator and do
    # cleanup. Try that now
    if num_samples == batch_size:
        try:
            next(dataloader)
        except StopIteration:
            pass
        else:
            raise ValueError
    assert not dataloader._working
    assert dataloader._batch_itr is None

    # check start of next epoch to ensure consistency
    X, y = next(dataloader)
    assert (y.numpy() == y0.numpy()).all()

    for column, x in X.items():
        x0 = X0.pop(column)
        assert (x.numpy() == x0.numpy()).all()
    assert len(X0) == 0

    dataloader.stop()
    assert not dataloader._working
    assert dataloader._batch_itr is None


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_mh_support(tmpdir, multihot_data, multihot_dataset, batch_size):
    data_itr = tf_dataloader.Loader(
        multihot_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    offsets = None
    idx = 0
    for X, y in data_itr:
        assert len(X) == 7
        n_samples = y.shape[0]

        for mh_name in ["Authors", "Reviewers", "Embedding"]:
            # assert (mh_name) in X
            array, offsets = X[f"{mh_name}__values"], X[f"{mh_name}__offsets"]
            offsets = offsets.numpy()
            array = array.numpy()
            lens = [0]
            cur = 0
            for x in multihot_data[mh_name][idx * batch_size : idx * batch_size + n_samples]:
                cur += len(x)
                lens.append(cur)
            assert (offsets == np.array(lens)).all()
            assert len(array) == max(lens)

        idx += 1
    assert idx == (3 // batch_size + 1)


@pytest.mark.parametrize("batch_size", [128, 256])
def test_validater(tmpdir, batch_size):
    n_samples = 10000
    rand = np.random.RandomState(0)

    df = make_df({"a": rand.randn(n_samples), "label": rand.randint(2, size=n_samples)})
    ds = Dataset(df)
    ds.schema["label"] = ds.schema["label"].with_tags(Tags.TARGET)

    dataloader = tf_dataloader.Loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    input_ = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    x = tf.keras.layers.Dense(128, "relu")(input_)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_, outputs=x)
    model.compile("sgd", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC()])

    validater = tf_dataloader.KerasSequenceValidater(dataloader)
    model.fit(dataloader, epochs=1, verbose=0, callbacks=[validater])

    predictions, labels = [], []
    for X, y_true in dataloader:
        y_pred = model(X)
        labels.extend(y_true.numpy())
        predictions.extend(y_pred.numpy()[:, 0])
    predictions = np.array(predictions)
    labels = np.array(labels)

    logs = {}
    validater.on_epoch_end(0, logs)
    auc_key = [i for i in logs if i.startswith("val_auc")][0]

    true_accuracy = (labels == (predictions > 0.5)).mean()
    print(true_accuracy)
    estimated_accuracy = logs["val_accuracy"]
    print(estimated_accuracy)
    assert np.isclose(true_accuracy, estimated_accuracy, rtol=0.1)

    true_auc = roc_auc_score(labels, predictions)
    estimated_auc = logs[auc_key]
    print(true_auc)
    print(estimated_auc)
    assert np.isclose(true_auc, estimated_auc, rtol=0.1)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("global_rank", [0, 1])
def test_multigpu_partitioning(dataset, batch_size, global_rank):
    data_loader = tf_dataloader.Loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        global_size=2,
        global_rank=global_rank,
    )
    indices = data_loader._indices_for_process()
    assert indices == [global_rank]


@pytest.mark.skipif(
    os.environ.get("NR_USER") is not None,
    reason="not working correctly in ci environment",
)
@pytest.mark.skipif(importlib.util.find_spec("horovod") is None, reason="needs horovod")
@pytest.mark.skipif(
    HAS_GPU and cupy and cupy.cuda.runtime.getDeviceCount() <= 1,
    reason="This unittest requires multiple gpu's to run",
)
def test_horovod_multigpu(tmpdir):
    json_sample = {
        "conts": {},
        "cats": {
            "genres": {
                "dtype": None,
                "cardinality": 50,
                "min_entry_size": 1,
                "max_entry_size": 5,
                "multi_min": 2,
                "multi_max": 4,
                "multi_avg": 3,
            },
            "movieId": {
                "dtype": None,
                "cardinality": 500,
                "min_entry_size": 1,
                "max_entry_size": 5,
            },
            "userId": {
                "dtype": None,
                "cardinality": 500,
                "min_entry_size": 1,
                "max_entry_size": 5,
            },
        },
        "labels": {"rating": {"dtype": None, "cardinality": 2}},
    }

    nvt = pytest.importorskip("nvtabular")
    datagen = pytest.importorskip("nvtabular.tools.data_gen")

    cols = datagen._get_cols_from_schema(json_sample)
    df_gen = datagen.DatasetGen(datagen.UniformDistro(), gpu_frac=0.0001)
    target_path = os.path.join(tmpdir, "input/")
    os.mkdir(target_path)
    df_files = df_gen.full_df_create(10000, cols, output=target_path)
    # process them
    cat_features = nvt.ColumnSelector(["userId", "movieId", "genres"]) >> nvt.ops.Categorify()
    ratings = nvt.ColumnSelector(["rating"]) >> nvt.ops.LambdaOp(
        lambda col: (col > 3).astype("int8")
    )
    output = cat_features + ratings
    proc = nvt.Workflow(output)
    target_path_train = os.path.join(tmpdir, "train/")
    os.mkdir(target_path_train)
    result_ds = proc.fit_transform(nvt.Dataset(df_files))
    schema = result_ds.schema
    schema["genres"] = schema["genres"]
    schema["rating"] = schema["rating"].with_tags(Tags.TARGET)
    result_ds.schema = schema
    result_ds.to_parquet(output_path=target_path_train, out_files_per_proc=5)
    # add new location
    target_path = os.path.join(tmpdir, "workflow/")
    os.mkdir(target_path)
    proc.save(target_path)
    curr_path = os.path.abspath(__file__)
    repo_root = os.path.relpath(os.path.normpath(os.path.join(curr_path, "../../../..")))
    hvd_wrap_path = os.path.join(repo_root, "merlin/dataloader/utils/tf/hvd_wrapper.sh")
    hvd_exam_path = os.path.join(repo_root, "merlin/dataloader/utils/tf/tf_trainer.py")
    with subprocess.Popen(
        [
            "horovodrun",
            "-np",
            "2",
            "-H",
            "localhost:2",
            "sh",
            hvd_wrap_path,
            "python",
            hvd_exam_path,
            "--dir_in",
            f"{tmpdir}",
            "--batch_size",
            "1024",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        process.wait()
        stdout, stderr = process.communicate()
        print(stdout, stderr)
        # if horovod failed to run because of environment issue, lets not fail the test here
        if b"horovod does not find an installed MPI." in stderr:
            pytest.skip("Skipping test because of horovod missing MPI")
        assert "Loss:" in str(stdout)


@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("cpu", [False, True] if HAS_GPU else [True])
def test_dataloader_schema(tmpdir, dataset, batch_size, cpu):
    with tf_dataloader.Loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    ) as data_loader:
        batch = data_loader.peek()

    columns = set(dataset.schema.column_names) - {"label"}
    assert set(batch[0]) == columns

    num_label_cols = batch[1].shape[1] if len(batch[1].shape) > 1 else 1
    assert num_label_cols == 1


def test_lazy_dataset_map():
    dataset_size = 100
    data_df = pd.DataFrame({"feature": np.random.randint(100, size=dataset_size)})
    dataset = Dataset(data_df)
    dataloader = tf_dataloader.Loader(dataset, batch_size=10)

    # since this is timing dependent, the first time a Dataloader works a bunch
    # of things are initialized - which takes 1.5s on my system
    # force a batch through so that the rest of the test can work even if this test
    # is called by itself
    next(dataloader)
    dataloader.stop()

    sleep_time_seconds = 0.5
    map_function_called = False

    def identity(x, y):
        nonlocal map_function_called
        time.sleep(sleep_time_seconds)
        map_function_called = True
        return (x, y)

    dataloader = dataloader.map(identity)

    elapsed_time_seconds = timeit.timeit(lambda: next(dataloader), number=1)

    assert map_function_called
    assert elapsed_time_seconds < 1


def test_keras_model_with_multiple_label_columns():
    df = make_df({"a": [0.1, 0.2, 0.3], "label1": [0, 1, 0], "label2": [1, 0, 0]})
    dataset = Dataset(df)
    dataset.schema["label1"] = dataset.schema["label1"].with_tags(Tags.TARGET)
    dataset.schema["label2"] = dataset.schema["label2"].with_tags(Tags.TARGET)

    loader = tf_dataloader.Loader(dataset, batch_size=1)

    inputs = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    outputs = tf.keras.layers.Dense(16, "relu")(inputs)
    output_1 = tf.keras.layers.Dense(2, activation="softmax", name="label1")(outputs)
    output_2 = tf.keras.layers.Dense(5, activation="softmax", name="label2")(outputs)
    # If we are using a Keras model and dataloader returns multiple labels,
    # `outputs` keys must match the multiple labels returned by the dataloader.
    model = tf.keras.Model(inputs=inputs, outputs={"label1": output_1, "label2": output_2})
    model.compile(
        optimizer="sgd", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    model.fit(loader, epochs=2)

    preds_model = model.predict({"a": tf.constant([0.1, 0.2, 0.3])})
    preds_loader = model.predict(loader)
    assert preds_model["label1"].shape == preds_loader["label1"].shape
    assert preds_model["label2"].shape == preds_loader["label2"].shape

    metrics = model.evaluate(loader, return_dict=True)
    assert "label1_accuracy" in metrics
    assert "label2_accuracy" in metrics
