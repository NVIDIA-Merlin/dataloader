import numpy as np
import tensorflow as tf

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags


class TFEmbeddingOperator(BaseOperator):
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

    def __init__(
        self,
        embeddings: np.ndarray,
        lookup_key: str = "id",
        embedding_name: str = "embeddings",
        id_lookup_table=None,
    ):
        self.embeddings = (
            embeddings if isinstance(embeddings, tf.Tensor) else tf.convert_to_tensor(embeddings)
        )
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.id_lookup_table = id_lookup_table

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        indices = transformable[self.lookup_key]
        if self.id_lookup_table is not None:
            indices = np.nonzero(np.in1d(self.id_lookup_table, indices))
        embeddings = tf.nn.embedding_lookup(self.embeddings, indices)
        transformable[self.embedding_name] = tf.squeeze(embeddings)
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
        indices = transformable[self.lookup_key]
        if self.id_lookup_table is not None:
            indices = np.in1d(self.id_lookup_table, indices)
        embeddings = self.embeddings[indices]
        transformable[self.embedding_name] = tf.squeeze(tf.convert_to_tensor(embeddings))
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
        if self.id_lookup is not None:
            ids_tensor = np.in1d(self.id_lookup, ids_tensor)
        embeddings = self.embeddings[ids_tensor]
        transformable[self.embedding_name] = tf.squeeze(tf.convert_to_tensor(embeddings))
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
