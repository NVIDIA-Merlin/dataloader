import numpy as np
import torch
from torch.nn import Embedding

from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import ColumnSchema, Schema, Tags


class TorchEmbeddingOperator(BaseOperator):
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
        self.embeddings = (
            embeddings
            if isinstance(embeddings, Embedding)
            else Embedding.from_pretrained(torch.FloatTensor(embeddings))
        )
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.id_lookup_table = id_lookup_table

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        indices = transformable[self.lookup_key]
        if self.id_lookup_table:
            indices = torch.Tensor(np.in1d(self.id_lookup_table, indices.cpu()))
        embeddings = self.embeddings(indices.cpu())
        transformable[self.embedding_name] = embeddings.to(indices.device)
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
                dtype=self.embeddings.weight.numpy().dtype,
                is_list=True,
                is_ragged=False,
            )
        )

        return Schema(col_schemas)


class Numpy_TorchEmbeddingOperator(BaseOperator):
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
        if self.id_lookup_table:
            indices = np.in1d(self.id_lookup_table, indices.cpu())
        embeddings = self.embeddings[np.in1d(self.embeddings[:, 0], indices.cpu())]
        transformable[self.embedding_name] = torch.from_numpy(embeddings[:, 1:]).to(indices.device)
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


class Numpy_Mmap_TorchEmbedding(BaseOperator):
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
        transform_function=None,
    ):
        self.embeddings = np.load(embedding_npz, mmap_mode="r")
        self.id_lookup = np.load(ids_lookup_npz) if ids_lookup_npz else None
        self.lookup_key = lookup_key
        self.embedding_name = embedding_name
        self.transform_function = transform_function or torch.from_numpy

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        indices = transformable[self.lookup_key]
        if self.id_lookup:
            indices = np.in1d(self.id_lookup[:, 0], indices)
        embeddings = self.embeddings[indices.cpu()]
        transformable[self.embedding_name] = self.transform_function(embeddings).to(indices.device)
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
