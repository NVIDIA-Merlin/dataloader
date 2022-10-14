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

# flake8: noqa
from merlin.loader.ops.embeddings.tf_embedding_op import (
    Numpy_Mmap_TFEmbedding,
    Numpy_TFEmbeddingOperator,
    TFEmbeddingOperator,
)
from merlin.loader.ops.embeddings.torch_embedding_op import (
    Numpy_Mmap_TorchEmbedding,
    Numpy_TorchEmbeddingOperator,
    TorchEmbeddingOperator,
)
