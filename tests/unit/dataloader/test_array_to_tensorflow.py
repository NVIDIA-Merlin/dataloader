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

from merlin.core.compat import tensorflow as tf
from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.schema import Tags

pytest.importorskip("tensorflow")
tf_dataloader = pytest.importorskip("merlin.dataloader.tensorflow")
loader = tf_dataloader.Loader


def test_array_dataloader_with_tensorflow():
    # Create dataset and loader
    df = make_df({"a": [0.1, 0.2, 0.3], "label": [0, 1, 0]})
    dataset = Dataset(df, cpu=True)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET)

    # Build model
    inputs = tf.keras.Input(name="a", dtype=tf.float32, shape=(1,))
    outputs = tf.keras.layers.Dense(16, "relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="softmax")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

    tf_loader = loader(dataset, batch_size=1)

    history_callback = model.fit(tf_loader, epochs=2, shuffle=False)

    assert len(history_callback.history["loss"]) == 2
    assert all(loss > 0.0 for loss in history_callback.history["loss"])

    preds_model = model.predict({"a": tf.constant([0.1, 0.2, 0.3])})
    preds_loader = model.predict(tf_loader)
    assert preds_model.shape == preds_loader.shape

    _ = model.evaluate(tf_loader)
