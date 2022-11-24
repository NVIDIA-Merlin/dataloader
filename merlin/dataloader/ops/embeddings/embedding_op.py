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


import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags


class EmbeddingOperator(BaseOperator):
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

    def __init__(
        self,
        embeddings: np.ndarray,
        lookup_key: str = "id",
        embedding_name: str = "embeddings",
        id_lookup_table=None,
    ):
        self.embeddings = self._load_embeddings(embeddings)
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.id_lookup_table = id_lookup_table

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        keys = transformable[self.lookup_key]
        indices = keys.cpu()
        if self.id_lookup_table is not None:
            indices = self._create_tensor(np.nonzero(np.in1d(self.id_lookup_table, indices)))
        embeddings = self._embeddings_lookup(indices)
        transformable[self.embedding_name] = self._format_embeddings(embeddings, keys)
        return transformable

    def _load_embeddings(self, embeddings):
        raise NotImplementedError("No logic supplied to load embeddings.")

    def _create_tensor(self, values):
        raise NotImplementedError("No logic supplied to create tensor.")

    def _embeddings_lookup(self, indices):
        raise NotImplementedError("No logic to look up embeddings with indices.")

    def _format_embeddings(self, embeddings, keys):
        raise NotImplementedError("No logic to format embeddings.")

    def _get_dtype(self, embeddings):
        raise NotImplementedError("No logic to retrieve dtype from embeddings.")

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Creates the output schema for this operator.

        Parameters
        ----------
        input_schema : Schema
            schema coming from ancestor nodes
        col_selector : ColumnSelector
            subselection of columns to apply to this operator
        prev_output_schema : Schema, optional
            the output schema of the previously executed operators, by default None

        Returns
        -------
        Schema
            Schema representing the correct output for this operator.
        """
        col_schemas = []
        for _, col_schema in input_schema.column_schemas.items():
            col_schemas.append(col_schema)
        col_schemas.append(
            ColumnSchema(
                name=self.embedding_name,
                tags=[Tags.CONTINUOUS],
                dtype=self._get_dtype(self.embeddings),
                is_list=True,
                is_ragged=False,
            )
        )

        return Schema(col_schemas)


class NumpyEmbeddingOperator(BaseOperator):
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

    def __init__(
        self,
        embeddings: np.ndarray,
        lookup_key: str = "id",
        embedding_name: str = "embeddings",
        id_lookup_table=None,
    ):
        self.embeddings = embeddings
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.id_lookup_table = id_lookup_table

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        keys = transformable[self.lookup_key]
        indices = keys.cpu()
        if self.id_lookup_table is not None:
            indices = np.in1d(self.id_lookup_table, indices)
        embeddings = self.embeddings[indices]
        # numpy_to_tensor
        transformable[self.embedding_name] = self._format_embeddings(embeddings, keys)
        return transformable

    def _format_embeddings(self, embeddings, keys):
        raise NotImplementedError("No logic to format embeddings.")

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        """Creates the output schema for this operator.

        Parameters
        ----------
        input_schema : Schema
            schema coming from ancestor nodes
        col_selector : ColumnSelector
            subselection of columns to apply to this operator
        prev_output_schema : Schema, optional
            the output schema of the previously executed operators, by default None

        Returns
        -------
        Schema
            Schema representing the correct output for this operator.
        """
        col_schemas = []
        for _, col_schema in input_schema.column_schemas.items():
            col_schemas.append(col_schema)
        col_schemas.append(
            ColumnSchema(
                name=self.embedding_name,
                tags=[Tags.CONTINUOUS],
                dtype=self.embeddings.dtype,
                is_list=True,
                is_ragged=False,
            )
        )

        return Schema(col_schemas)


class MmapNumpyTorchEmbedding(NumpyEmbeddingOperator):
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

    def __init__(
        self,
        embedding_npz,
        ids_lookup_npz=None,
        lookup_key="id",
        embedding_name="embeddings",
    ):
        embeddings = np.load(embedding_npz, mmap_mode="r")
        id_lookup = np.load(ids_lookup_npz) if ids_lookup_npz else None
        super().__init__(
            embeddings,
            lookup_key=lookup_key,
            embedding_name=embedding_name,
            id_lookup_table=id_lookup,
        )
