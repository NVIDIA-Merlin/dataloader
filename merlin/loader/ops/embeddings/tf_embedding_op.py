import numpy as np
import tensorflow as tf

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags


class TFEmbeddingOperator(BaseOperator):
    def __init__(self, embeddings, lookup_key: str = "id", embedding_name: str = "embeddings"):
        self.embeddings = (
            embeddings if isinstance(embeddings, tf.Tensor) else tf.convert_to_tensor(embeddings)
        )
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        embeddings = tf.nn.embedding_lookup(self.embeddings, transformable[self.lookup_key])
        transformable[self.embedding_name] = embeddings
        return transformable

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
        col_schemas = []
        for _, col_schema in input_schema.column_schemas.items():
            col_schemas.append(col_schema)
        col_schemas.append(
            ColumnSchema(
                name=self.embedding_name,
                tags=[Tags.CONTINUOUS],
                dtype=self.embeddings.dtype.as_numpy_dtype,
                is_list=True,
                is_ragged=False,
            )
        )

        return Schema(col_schemas)


class Numpy_TFEmbeddingOperator(BaseOperator):
    def __init__(self, embeddings, lookup_key: str = "id", embedding_name: str = "embeddings"):
        self.embeddings = embeddings
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        ids_tensor = transformable[self.lookup_key]
        embeddings = self.embeddings[np.in1d(self.embeddings[:, 0], ids_tensor)]
        transformable[self.embedding_name] = tf.convert_to_tensor(embeddings[:, 1:])
        return transformable

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
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


class Numpy_Mmap_TFEmbedding(BaseOperator):
    def __init__(
        self,
        embedding_npz,
        ids_lookup_npz=None,
        lookup_key="id",
        embedding_name="embeddings",
        transform_function=None,
    ):
        self.embeddings = np.load(embedding_npz, mmap_mode="r")
        self.id_lookup = np.load(ids_lookup_npz) if ids_lookup_npz else None
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.transform_function = tf.convert_to_tensor

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        ids_tensor = transformable[self.lookup_key]
        if self.id_lookup:
            ids_tensor = np.in1d(self.id_lookup[:, 0], ids_tensor)
        embeddings = self.embeddings[ids_tensor]
        if self.transform_function:
            transformable[self.embedding_name] = self.transform_function(embeddings)
        else:
            transformable[self.embedding_name] = embeddings
        return transformable

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: Schema = None,
    ) -> Schema:
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
