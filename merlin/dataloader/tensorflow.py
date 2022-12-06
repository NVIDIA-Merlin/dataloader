#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
import contextlib
import logging

from merlin.dataloader.loader_base import LoaderBase
from merlin.dataloader.tf_utils import configure_tensorflow

from_dlpack = configure_tensorflow()
LOG = logging.getLogger("dataloader")

# tf import must happen after config to restrict memory use
import tensorflow as tf  # noqa

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,redundant-keyword-arg


class Loader(tf.keras.utils.Sequence, LoaderBase):
    """
    Infinite generator used to asynchronously iterate through CSV or Parquet
    dataframes on GPU by leveraging an `merlin.io.Dataset`.

    This lazily loads merlin.io.Dataset objects  and outputs tabular dictionaries of TensorFlow
    Tensors via `dlpack <https://github.com/dmlc/dlpack>`_. Useful for training tabular models
    built in Keras and trained via
    `tf.keras.Model.fit <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.

    The data loading scheme is implemented by loading, preprocessing, and
    batching data in an asynchronous thread. The amount of randomness in
    shuffling is controlled by the `buffer_size` and `parts_per_chunk`
    kwargs. At load time, sub-chunks of data with size controlled by
    `buffer_size` are loaded from random partitions in the dataset,
    and `parts_per_chunk` of them are concatenated into a single chunk,
    shuffled, and split into batches. This means that each chunk has
    `buffer_size*parts_per_chunk` rows, and due to the asynchronous
    nature of the dataloader that means there are, including the batch
    being processed by your network, `3*buffer_*parts_per_chunk`
    rows of data in GPU memory at any given time. This means that
    for a fixed memory budget, using more `parts_per_chunk` will
    come at the expense of smaller `buffer_size`, increasing the number
    of reads and reducing throughput. The goal should be to maximize the
    total amount of memory utilized at once without going OOM and with
    the fewest number of reads to meet your epoch-level randomness needs.

    An important thing to note is that TensorFlow's default behavior
    is to claim all GPU memory for itself at initialziation time,
    which leaves none for this class to load or preprocess data.
    As such, we attempt to configure TensorFlow to restrict
    its memory allocation on a given GPU using the environment variables
    `TF_MEMORY_ALLOCATION` and `TF_VISIBLE_DEVICE`. If `TF_MEMORY_ALLOCATION < 1`,
    it will be assumed that this refers to a fraction of free GPU
    memory on the given device. Otherwise, it will refer to an explicit
    allocation amount in MB. `TF_VISIBLE_DEVICE` should be an integer GPU
    index.

    Iterator output is of the form `(dict(features), list(labels))`,
    where each element of the features dict is a
    `feature_name: feature_tensor` and each element of the labels
    list is a tensor, and all tensors are of shape `(batch_size, 1)`.

    We're implementing the keras Sequence interface because using the
    loader as an iterator with keras has a limitation of only being able
    to run for one epoch.

    Parameters
    -------------
    dataset: merlin.io.Dataset
        The dataset to load
    batch_size: int
        Number of rows to yield at each iteration
    shuffle: bool, default True
        Whether to shuffle chunks of batches before iterating through them.
    seed_fn: callable
        Function used to initialize random state
    parts_per_chunk: int
        Number of dataset partitions with size dictated by `buffer_size`
        to load and concatenate asynchronously. More partitions leads to
        better epoch-level randomness but can negatively impact throughput
    global_size: int, optional
        When doing distributed training, this indicates the number of total processes that are
        training the model.
    global_rank:
        When doing distributed training, this indicates the local rank for the current process.
    drop_last: bool, default False
        Whether or not to drop the last batch in an epoch. This is useful when you need to
        guarantee that each batch contains exactly `batch_size` rows - since the last batch
        will usually contain fewer rows.
    """

    _use_nnz = True

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        seed_fn=None,
        parts_per_chunk=1,
        global_size=None,
        global_rank=None,
        drop_last=False,
        transforms=None,
        device=None,
    ):
        LoaderBase.__init__(
            self,
            dataset,
            batch_size,
            shuffle=shuffle,
            seed_fn=seed_fn,
            parts_per_chunk=parts_per_chunk,
            global_size=global_size,
            global_rank=global_rank,
            drop_last=drop_last,
            transforms=transforms,
            device=device,
        )
        self._map_fns = []

    def __len__(self):
        """Number of batches in the Sequence.

        Note: This also resets the loader state.
              Required because of the calls to `__getitem__`
              from keras prior to the start of the main loop
              through the loader.
        """
        LoaderBase.stop(self)
        return LoaderBase.__len__(self)

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Note: This returns the next batch in the iterator.
              Not the batch at position `index`.
              This is because the dataloader is implemented as an iterator and
              don't currently support fetching a batch by index.
        """
        return LoaderBase.__next__(self)

    def on_epoch_end(self):
        "Method called at the end of every epoch."
        self.stop()

    def map(self, fn):
        """
        Applying a function to each batch.

        This can for instance be used to add `sample_weight` to the model.
        """
        self._map_fns.append(fn)

        return self

    @contextlib.contextmanager
    def _get_device_ctx(self, dev):
        # with tf.device("/device:GPU:{}".format(dev)) as tf_device:
        #     # tf.device changes the cupy cuda device, which breaks us on multigpu
        #     # force cupy to still use the device we expect
        #     cupy.cuda.Device(dev).use()
        #     yield tf_device
        # commenting out since device statements cause
        # RuntimeErrors when exiting if two dataloaders
        # are running at once (e.g. train and validation)
        if dev != "cpu":
            yield tf.device("/GPU:" + str(dev))
        else:
            # https://www.tensorflow.org/guide/gpu#manual_device_placement
            yield tf.device("/device:CPU:0")

    def _split_fn(self, tensor, idx, axis=0):
        return tf.split(tensor, idx, axis=axis)

    def _tensor_split(self, tensor, idx, axis=0):
        """
        Same function as above but need this method
        for api match.
        """
        return tf.split(tensor, idx, axis=axis)

    @property
    def _LONG_DTYPE(self):
        return tf.int64

    @property
    def _FLOAT32_DTYPE(self):
        return tf.float32

    def _unpack(self, gdf):
        if hasattr(gdf, "shape"):
            return tf.convert_to_tensor(gdf)
        return from_dlpack(gdf)

    def _to_tensor(self, gdf):
        if gdf.empty:
            return

        # checks necessary because of this bug
        # https://github.com/tensorflow/tensorflow/issues/42660
        if len(gdf.shape) == 1 or gdf.shape[1] == 1:
            dlpack = self._pack(gdf)
        elif gdf.shape[0] == 1:
            dlpack = self._pack(gdf.values[0])
        else:
            dlpack = self._pack(gdf.values.T)

        # catch error caused by tf eager context
        # not being initialized
        try:
            x = self._unpack(dlpack)
        except AssertionError:
            tf.random.uniform((1,))
            x = self._unpack(dlpack)
        # if rank is already two it is  already in list format
        if gdf.shape[0] == 1 and not tf.rank(x) == 2:
            # batch size 1 so got squashed to a vector
            x = tf.expand_dims(x, 0)
        elif len(gdf.shape) == 1 or len(x.shape) == 1:
            # sort of a generic check for any other
            # len(shape)==1 case, could probably
            # be more specific
            x = tf.expand_dims(x, -1)
        elif gdf.shape[1] > 1:
            # matrix which means we had to transpose
            # for the bug above, so untranspose
            x = tf.transpose(x)
        return x

    def _pull_values_offsets(self, values_offset):
        """
        values_offset is either a tuple (values, offsets) or just values.
        Values is a tensor.
        This method is used to turn a tensor into its sparse representation
        """
        # pull_values_offsets, return values offsets diff_offsets
        diff_offsets = None
        if isinstance(values_offset, tuple):
            values = tf.reshape(values_offset[0], [-1])
            diff_offsets = tf.cast(tf.reshape(values_offset[1], [-1]), dtype=tf.int64)
            offsets = tf.math.cumsum(diff_offsets)
        else:
            values = tf.reshape(values_offset, [-1])
            offsets = tf.arange(tf.shape(values)[0], dtype=tf.int64)
            diff_offsets = offsets[1:] - offsets[:-1]
        num_rows = len(offsets)
        return values, offsets, diff_offsets, num_rows

    def _get_max_seq_len(self, diff_offsets):
        # get_max_seq_len, return int
        return int(tf.math.reduce_max(diff_offsets))

    def _get_indices(self, offsets, diff_offsets):
        # Building the indices to reconstruct the sparse tensors
        row_ids = tf.range(len(offsets), dtype=tf.int64)

        row_ids_repeated = tf.repeat(row_ids, diff_offsets)
        row_offset_repeated = tf.repeat(offsets, diff_offsets)
        col_ids = tf.range(len(row_offset_repeated), dtype=tf.int64) - row_offset_repeated
        indices = tf.concat(
            values=[tf.expand_dims(row_ids_repeated, -1), tf.expand_dims(col_ids, -1)],
            axis=1,
        )
        return indices

    def _get_sparse_tensor(self, values, indices, num_rows, seq_limit):
        sparse_tensor = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=[num_rows, seq_limit]
        )
        return sparse_tensor

    def _build_sparse_tensor(
        self, values, offsets, diff_offsets, num_rows, seq_limit, sparse_as_dense
    ):
        ragged = tf.RaggedTensor.from_row_lengths(values=values, row_lengths=diff_offsets)
        tensor = tf.RaggedTensor.from_tensor(ragged.to_tensor(shape=[None, seq_limit])).to_sparse()
        if sparse_as_dense:
            tensor = tf.sparse.to_dense(tensor)
        return tensor

    def _handle_tensors(self, tensors, tensor_names):
        to_return = super()._handle_tensors(tensors, tensor_names)

        for map_fn in self._map_fns:
            to_return = map_fn(*to_return)

        return to_return

    def _cast_to_numpy_dtype(self, dtype):
        """
        Get the numpy dtype from the framework dtype.
        """
        return dtype.as_numpy_dtype()


class KerasSequenceValidater(tf.keras.callbacks.Callback):
    # TODO: document
    _supports_tf_logs = True

    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_epoch_end(self, epoch, logs=None):
        """Callback that runs at the end of an epoch.

        Parameters
        ----------
        epoch : int
            Integer representing the current epoch.
        logs : Dict, optional
            dictionary of logs collected, to be added to, by default None

        Returns
        -------
        logs
            Dictionary with results from callback.
        """
        logs = logs if logs is not None else {}
        for X, y_true in self.dataloader:
            y_pred = self.model(X)

            # TODO: how do we want to handle the multi-output case?
            for metric in self.model.metrics:
                metric.update_state(y_true, y_pred)

        set_logs = {}
        for metric in self.model.metrics:
            set_logs[f"val_{metric.name}"] = metric.result().numpy()
        logs.update(set_logs)
        print(set_logs)
        return logs
