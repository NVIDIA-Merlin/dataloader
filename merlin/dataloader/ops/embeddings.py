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
from typing import Optional, Union

import numpy as np

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.table import Device, TensorColumn


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
        embeddings: Union[np.ndarray, str],
        lookup_key: str = "id",
        embedding_name: str = "embeddings",
        id_lookup_table: Optional[Union[np.ndarray, str]] = None,
        mmap=False,
    ):
        if mmap:
            embeddings = np.load(embeddings, mmap_mode="r")
            id_lookup_table = np.load(id_lookup_table) if id_lookup_table else None
        self.embeddings = embeddings
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.id_lookup_table = id_lookup_table

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        keys = transformable[self.lookup_key]
        indices = keys.cpu().values
        if self.id_lookup_table is not None:
            indices = np.in1d(self.id_lookup_table, indices)
        embeddings = self.embeddings[indices]
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
        col_schemas.append(
            ColumnSchema(
                name=self.embedding_name,
                tags=[Tags.EMBEDDING],
                dtype=self.embeddings.dtype,
                dims=id_schema.shape.as_tuple + (embedding_dim,),
            )
        )

        return Schema(col_schemas)
