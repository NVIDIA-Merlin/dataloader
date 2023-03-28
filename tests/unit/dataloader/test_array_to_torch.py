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

from merlin.core.compat import torch as th
from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.schema import Tags

torch_loader = pytest.importorskip("merlin.dataloader.frameworks.torch")
loader = torch_loader.TorchArrayDataloader


def test_array_dataloader_with_torch():
    # Create dataset and loader
    df = make_df({"a": [[0.1], [0.2], [0.3]], "label": [[0], [1], [0]]})
    dataset = Dataset(df, cpu=True)
    dataset.schema["label"] = dataset.schema["label"].with_tags(Tags.TARGET).with_shape((3, 1))
    dataset.schema["a"] = dataset.schema["a"].with_shape((3, 1))

    class NeuralNetwork(th.nn.Module):
        layers = th.nn.Sequential(th.nn.Linear(1, 16), th.nn.Linear(16, 1), th.nn.Softmax(dim=0))

        def forward(self, batch):
            if isinstance(batch, loader):
                batches = [list(tup[0].values())[0] for tup in list(iter(batch))]
                x_tens = th.cat(tuple(batches), 0)
            elif isinstance(batch, tuple):
                x, _ = batch
                tups_x = tuple(x.values())
                x_tens = th.cat(tups_x, 1)
            elif isinstance(batch, th.Tensor):
                x_tens = batch
            y_preds = self.layers(x_tens.to(th.float32))
            return y_preds

    loss_fn = th.nn.BCELoss(reduction="sum")

    # Build model
    model = NeuralNetwork()

    th_loader = loader(dataset, batch_size=1)

    for batch in th_loader:
        y_pred = model(batch)
        loss = loss_fn(y_pred, batch[1].to(th.float32))
        model.zero_grad()
        loss.backward()

    # assert len(history_callback.history["loss"]) == 2
    # assert all(loss > 0.0 for loss in history_callback.history["loss"])

    preds_model = model(th.tensor([[0.1], [0.2], [0.3]], dtype=th.float32))
    preds_loader = model(th_loader)
    assert preds_model.shape == preds_loader.shape
