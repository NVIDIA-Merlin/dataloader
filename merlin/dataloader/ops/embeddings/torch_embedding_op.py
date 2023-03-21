#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import torch
from torch.nn import Embedding

from merlin.dataloader.ops.embeddings.embedding_op import (
    EmbeddingOperator,
    MmapNumpyTorchEmbedding,
    NumpyEmbeddingOperator,
)


class TorchEmbeddingOperator(EmbeddingOperator):
    """Create an operator that will apply a torch embedding table to supplied indices.
    This operator allows the user to supply an id lookup table if the indices supplied
    via the id_lookup_table.

    Parameters
    ----------
    embeddings : np.ndarray
        numpy ndarray representing embedding values
    lookup_key : str, optional
        the name of the column that will be used as indices, by default "id"
    embedding_name : str, optional
        name of new column of embeddings, added to output, by default "embeddings"
    id_lookup_table : np.array, optional
        numpy array of values that represent embedding indices, by default None
    """

    def _load_embeddings(self, embeddings):
        return (
            embeddings
            if isinstance(embeddings, Embedding)
            else Embedding.from_pretrained(torch.FloatTensor(embeddings))
        )

    def _create_tensor(self, values):
        return torch.Tensor(values).to(torch.int32)

    def _embeddings_lookup(self, indices):
        return self.embeddings(indices)

    def _format_embeddings(self, embeddings, keys):
        return torch.squeeze(embeddings.to(keys.device))

    def _get_dtype(self, embeddings):
        return embeddings.weight.numpy().dtype


class Torch_NumpyEmbeddingOperator(NumpyEmbeddingOperator):
    """Create an embedding table from supplied embeddings to add embedding entry
    to records based on supplied indices. Support for indices lookup table is available.
    Embedding table is stored in host memory.

    Parameters
    ----------
    embeddings : np.ndarray
        numpy ndarray representing embedding values
    lookup_key : str, optional
        the name of the column that will be used as indices, by default "id"
    embedding_name : str, optional
        name of new column of embeddings, added to output, by default "embeddings"
    id_lookup_table : np.array, optional
        numpy array of values that represent embedding indices, by default None
    """

    def _format_embeddings(self, embeddings, keys):
        return torch.from_numpy(embeddings).to(keys.device)


class Torch_MmapNumpyTorchEmbedding(MmapNumpyTorchEmbedding):
    """Operator loads numpy embedding table from file using memory map to be used to create
    torch embedding representations. This allows for larger than host memory embedding
    tables to be used for embedding lookups. The only limit to the size is what fits in
    storage, preferred storage device is SSD for faster lookups.

    Parameters
    ----------
    embedding_npz : numpy ndarray file
        file holding numpy ndarray representing embedding table
    ids_lookup_npz : numpy array file, optional
        file holding numpy array of values that represent embedding indices, by default None
    lookup_key : str, optional
        the name of the column that will be used as indices, by default "id"
    embedding_name : str, optional
        name of new column of embeddings, added to output, by default "embeddings"
    transform_function : _type_, optional
        function that will transform embedding from numpy to torch, by default None
    """

    def _format_embeddings(self, embeddings, keys):
        return torch.from_numpy(embeddings).to(keys.device)
