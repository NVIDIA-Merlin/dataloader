# [Merlin Dataloader](https://github.com/NVIDIA-Merlin/dataloader)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/merlin-dataloader)
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-dataloader.svg)](https://pypi.python.org/pypi/merlin-dataloader/)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/dataloader)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/dataloader/stable/README.html)

The merlin-dataloader lets you quickly train recommender models for TensorFlow, PyTorch and JAX. It eliminates the biggest bottleneck in training recommender models, by providing GPU optimized dataloaders that read data directly into the GPU, and then do a 0-copy transfer to TensorFlow and PyTorch using [dlpack](https://github.com/dmlc/dlpack).

The benefits of the Merlin Dataloader include:

- Over 10x speedup over native framework dataloaders
- Handles larger than memory datasets
- Per-epoch shuffling
- Distributed training

## Installation

Merlin-dataloader requires Python version 3.8+. Additionally, GPU support requires CUDA 11.0+.

To install using Conda into an existing environment

```
conda install -c nvidia merlin-dataloader
```

To create a new conda environment with cuDF and TensorFlow installed with mamba:

```
mamba create -n merlin-dataloader -c nvidia -c rapidsai -c conda-forge \
    merlin-dataloader=23.06 \
    tensorflow=2.12 \
    python=3.10 \
    cudf=23.04 \
    cudatoolkit=11.8
```

To install from PyPi:

```
pip install merlin-dataloader
```

There are also [docker containers on NGC](https://nvidia-merlin.github.io/Merlin/stable/containers.html) with the merlin-dataloader and dependencies included on them

## Basic Usage

```python
# Get a merlin dataset from a set of parquet files
import merlin.io
dataset = merlin.io.Dataset(PARQUET_FILE_PATHS, engine="parquet")

# Create a Tensorflow dataloader from the dataset, loading 65K items
# per batch
from merlin.dataloader.tensorflow import Loader
loader = Loader(dataset, batch_size=65536)

# Get a single batch of data. Inputs will be a dictionary of columnname
# to TensorFlow tensors
inputs, target = next(loader)

# Train a Keras model with the dataloader
model = tf.keras.Model( ... )
model.fit(loader, epochs=5)
```
