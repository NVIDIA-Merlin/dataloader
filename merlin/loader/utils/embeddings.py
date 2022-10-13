import gc

import numpy as np
from npy_append_array import NpyAppendArray

from merlin.core import dispatch


def build_embeddings_from_pq(
    df_paths, embedding_filename="embeddings.npy", lookup_filename="lookup_ids"
):
    df_lib = dispatch.get_lib()
    with NpyAppendArray(embedding_filename) as nf:
        with NpyAppendArray(lookup_filename) as lf:
            for path in df_paths:
                rows = df_lib.read_parquet(path)
                numpy_rows = rows.to_numpy()
                indices = np.ascontiguousarray(numpy_rows[:, 0])
                vectors = np.ascontiguousarray(numpy_rows[:, 1:])
                lf.append(indices)
                nf.append(vectors)
                del rows
                del numpy_rows
                del indices
                del vectors
                gc.collect()
    return embedding_filename, lookup_filename
