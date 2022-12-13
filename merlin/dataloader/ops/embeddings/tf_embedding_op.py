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

import tensorflow as tf

from merlin.dataloader.ops.embeddings.embedding_op import (
    EmbeddingOperator,
    MmapNumpyTorchEmbedding,
    NumpyEmbeddingOperator,
)


class TFEmbeddingOperator(EmbeddingOperator):
    """Create an operator that will apply a tf embedding table to supplied indices.
    This operator allows the user to supply an id lookup table if the indices supplied
    via the id_lookup_table. Embedding table is stored in host memory.

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
        return embeddings if isinstance(embeddings, tf.Tensor) else tf.convert_to_tensor(embeddings)

    def _create_tensor(self, values):
        return values

    def _embeddings_lookup(self, indices):
        return tf.nn.embedding_lookup(self.embeddings, indices)

    def _format_embeddings(self, embeddings, keys):
        return tf.squeeze(embeddings)

    def _get_dtype(self, embeddings):
        return embeddings.dtype.as_numpy_dtype


class TF_NumpyEmbeddingOperator(NumpyEmbeddingOperator):
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
        return tf.squeeze(tf.convert_to_tensor(embeddings))


class TF_MmapNumpyTorchEmbedding(MmapNumpyTorchEmbedding):
    """Operator loads numpy embedding table from file using memory map to be used to create
    tensorflow embedding representations. This allows for larger than host memory embedding
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
        return tf.squeeze(tf.convert_to_tensor(embeddings))
