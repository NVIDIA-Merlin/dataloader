from typing import Union

import numpy as np

from merlin.core.compat import cupy
from merlin.core.protocols import Transformable
from merlin.dag import BaseOperator
from merlin.dag.selector import ColumnSelector
from merlin.schema import Schema
from merlin.table import Device, TensorTable


class Padding(BaseOperator):
    """Create an operator that will apply a embedding table to supplied indices.
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

    def __init__(self, padding_size: int, padding_value: Union[int, float]):
        self.padding_size = padding_size
        self.padding_value = padding_value

    def transform(
        self, col_selector: ColumnSelector, transformable: Transformable
    ) -> Transformable:
        to_pad = transformable[col_selector.names]
        if not isinstance(transformable, TensorTable):
            to_pad = TensorTable.from_df(to_pad)
        for col_name, col in to_pad.items():
            col._values = pad_put_zeros(col, self.padding_size, self.padding_value)
            col._offsets = None
        return type(transformable)(to_pad)

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
        for col_name, col_schema in input_schema.column_schemas.items():
            old_shape = input_schema[col_name].dtype.shape.as_tuple
            col_schemas.append(input_schema[col_name].with_shape((old_shape[0], self.padding_size)))

        return Schema(col_schemas)


def pad_put_zeros(column, padding_size, padding_val):
    # account for zero prepend
    array_lib = cupy if column.device == Device.GPU else np
    num_rows = len(column.offsets) - 1
    zeros = array_lib.zeros((num_rows, padding_size)).flatten()
    row_lengths = column.offsets[1:] - column.offsets[:-1]
    row_ranges = []
    starts = array_lib.arange(num_rows) * padding_size
    ends = starts + row_lengths
    for idx, offset in enumerate(column.offsets[:-1]):
        row_ranges.extend(array_lib.arange(int(starts[idx]), int(ends[idx])))
    array_lib.put(zeros, row_ranges, column.values)
    zeros = array_lib.reshape(zeros, (num_rows, padding_size))
    return zeros
