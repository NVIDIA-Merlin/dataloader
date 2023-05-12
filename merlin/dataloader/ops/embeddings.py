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
import os
from typing import Optional, Union

import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.table import Device, TensorColumn


class EmbeddingOperator(BaseOperator):
    """Create an operator that will apply an embedding table to supplied indices.

    An id lookup table for the embeddings can be supplied with the argument `id_lookup_table`.

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
    mmap : bool, default False
        When loading embeddings from a file, specify whether we should memory map the file
        This is useful for accessing a large file without reading the entire file into memory.
    unknown_value : Union[float, int, np.ndarray]
        If an embedding index is not found.
        Specifies the value should we return for the corresponding embedding.
    """

    def __init__(
        self,
        embeddings: Union[np.ndarray, str],
        lookup_key: str = "id",
        embedding_name: str = "embeddings",
        id_lookup_table: Optional[Union[np.ndarray, str]] = None,
        mmap: bool = False,
        unknown_value: Union[float, int, np.ndarray] = 0,
    ):
        if isinstance(embeddings, (str, os.PathLike)):
            mmap_mode = "r" if mmap else None
            embeddings = np.load(embeddings, mmap_mode=mmap_mode)
        elif isinstance(embeddings, np.ndarray):
            pass
        else:
            raise ValueError(
                f"Unsupported type '{type(embeddings)}' passed to argument `embeddings` "
                f"of '{type(self).__name__}'. "
                "Expected either a numpy.ndarray "
                "or a (string or pathlike object corresponding to a numpy file) "
                "containing embeddings. "
            )
        self.embeddings = embeddings

        embedding_index_mapping = None
        if isinstance(id_lookup_table, (str, os.PathLike)):
            _ids = np.load(id_lookup_table)
            embedding_index_mapping = self._get_embedding_index_mapping(_ids)
        elif isinstance(id_lookup_table, np.ndarray):
            _ids = id_lookup_table
            embedding_index_mapping = self._get_embedding_index_mapping(_ids)
        elif id_lookup_table is None:
            pass
        else:
            raise ValueError(
                f"Unsupported type '{type(id_lookup_table)}' passed to argument `id_lookup_table` "
                f"of '{type(self).__name__}'. "
                "Expected either a numpy.ndarray "
                "or a (string or pathlike object corresponding to a numpy file) "
                "containing the IDs that correspond to the embeddings. "
            )
        self.embedding_index_mapping = embedding_index_mapping

        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.unknown_value = unknown_value

    def _get_embedding_index_mapping(self, ids):
        expected_ids_shape = (self.embeddings.shape[0],)
        assert ids.shape == expected_ids_shape, (
            "IDs provided must match the number of embeddings. "
            f"Expected IDs with shape {expected_ids_shape} "
            f"Received IDs with shape: {ids.shape} "
            f"Embeddings shape: {self.embeddings.shape} "
        )
        id_to_index_mapping = dict(zip(ids, range(len(ids))))
        return id_to_index_mapping

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        keys = transformable[self.lookup_key]
        indices = keys.cpu().values

        if self.embedding_index_mapping is not None:
            indices = np.array([self.embedding_index_mapping.get(_id, -1) for _id in indices])

        embeddings = self.embeddings[indices]

        # set unknown embedding to zero
        for idx in np.ndindex(indices.shape):
            embedding_index = indices[idx]
            if embedding_index == -1:
                embeddings[idx] = self.unknown_value

        embeddings_col = TensorColumn(embeddings, offsets=keys.cpu().offsets)
        transformable[self.embedding_name] = (
            embeddings_col.gpu() if keys.device == Device.GPU else embeddings_col
        )
        return transformable

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
        id_schema = input_schema.column_schemas[self.lookup_key]
        embedding_dim = self.embeddings.shape[1]

        new_tags = [Tags.EMBEDDING]
        propagated_tags = [
            Tags.LIST,
            Tags.SEQUENCE,
            Tags.USER,
            Tags.ITEM,
            Tags.CONTEXT,
            Tags.SESSION,
        ]
        for tag in propagated_tags:
            if tag in id_schema.tags:
                new_tags.append(tag)
        col_schemas.append(
            ColumnSchema(
                name=self.embedding_name,
                tags=new_tags,
                dtype=self.embeddings.dtype,
                dims=id_schema.shape.as_tuple + (embedding_dim,),
            )
        )

        return Schema(col_schemas)
